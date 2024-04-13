
import datetime
import hashlib
import argparse
import json
import os
import traceback
from typing import Dict, List
import torch
import random

import transformers
import datasets

# Create the goal function using the model
from textattack.goal_functions import UntargetedClassification
from textattack.goal_functions import TextToTextGoalFunction
from textattack.goal_functions import MinimizeBleu
from dtcover.constraints.pre_transformation.multi_choice_options import MultiChoiceOptionsModification
from dtcover.constraints.pre_transformation.multi_choice_instruction import MultiChoiceInstructionModification
from dtcover.core import inject_module, recover_module
from dtcover.cover_attack import attack_dataset
from dtcover.data.mmlu import OPTIONS, format_example, format_subject, gen_prompt
from dtcover.goal_functions.classification.next_token_classification import UntargetedNextTokenClassification
from dtcover.goal_functions.classification.mcq import UntargetedMultipleChoiceQuestionClassification
from dtcover.inference import lm_forward
from dtcover.injection import get_injection_module
from dtcover.llm_attacker import LLMAttacker

from dtcover.metrics.ncov_mlp import NCovMlp
from dtcover.metrics.ncov_pair_atten import NCovPairAtten
from dtcover.metrics.ncov_pos_atten import NCovPosAtten
from dtcover.models.wrappers.huggingface_model_for_causal_lm_wraper import HuggingFaceModelForCausalLMWrapper

# Import the dataset
from textattack.datasets import HuggingFaceDataset

from textattack.search_methods import GreedySearch, BeamSearch
from textattack.constraints.pre_transformation import (
    RepeatModification,
    StopwordModification,
    MaxNumWordsModified,
)
from textattack import Attack

from tqdm import tqdm  # tqdm provides us a nice progress bar.
from textattack.loggers import CSVLogger  # tracks a dataframe for us.
from textattack.attack_results import SuccessfulAttackResult, AttackResult
from textattack import Attacker
from textattack import AttackArgs
from textattack.transformations.word_swaps import WordSwapRandomCharacterDeletion
from textattack.transformations.word_swaps import WordSwapRandomCharacterInsertion
from textattack.transformations.word_swaps import WordSwapNeighboringCharacterSwap
# from textattack.transformations.word_swaps import WordSwapQWERTY
from dtcover.transformations.multi_transformation import MultiTransformation

from dtcover.monitors.reduce_monitor import ReduceMonitor
from dtcover.monitors.reduce_ops import maxmin_fn

from dtcover.transformations.word_swaps.word_swap_qwerty import WordSwapQWERTY
from dtcover.transformations.word_swaps.hotflip_based import WordSwapHotFlipBased
from dtcover.transformations.word_swaps.fgsm_based import WordSwapFGSMBased
from dtcover.transformations.word_swaps.coverage_fgsm_based import WordSwapCoverageFGSMBased
from dtcover.transformations.word_swaps.coverage_hotflip_based import WordSwapCoverageHotFlipBased
from dtcover.transformations.word_swaps.banana import BananaWordSwap

from dtcover.metrics import CovKind, CovType

from joblib import Memory

from dtcover.utils import get_token_num
location = '~/.cache/dtcover_cache'
memory = Memory(location, verbose=0)

import logging
def get_logger(name: str, path: str, to_stdout=False):
    logger = logging.getLogger(name)

    for handler in logger.root.handlers[:]:
        logger.root.removeHandler(handler)

    formatter = logging.Formatter('[%(asctime)s] [%(name)s %(levelname)s]: %(message)s')

    if os.path.exists(path):
        os.remove(path)
    fh = logging.FileHandler(path, mode='w+', encoding='utf-8')

    fh.setFormatter(formatter)
    logger.addHandler(fh)
    
    if to_stdout:
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    logger.setLevel(logging.DEBUG)
    return logger

# @memory.cache
def load_dataset_all(dataset_path: str, split: str):
    datasets_configs = datasets.get_dataset_config_names(dataset_path)
    if "all" in datasets_configs:
        datasets_configs.remove("all")
    if "auxiliary_train" in datasets_configs:
        datasets_configs.remove("auxiliary_train")
    
    # datasets_configs = [datasets_configs[0]]
    all_datasets = []
    for each_config in tqdm(datasets_configs):
        all_datasets.append(datasets.load_dataset(dataset_path, each_config, split=split, trust_remote_code=True))
    all_dataset = datasets.concatenate_datasets(all_datasets)

    return all_dataset

@memory.cache
def load_mmlu_dataset(dataset_path: str, devsplit: str, testsplit: str):
    dev_dataset = load_dataset_all(dataset_path, devsplit)
    test_dataset = load_dataset_all(dataset_path, testsplit)
    # dev_dataset = dev_dataset.shuffle(seed=42).select(range(len(dev_dataset) // 10))
    # test_dataset = test_dataset.shuffle(seed=42).select(range(len(test_dataset) // 10))
    
    print(len(dev_dataset), len(test_dataset))
    return dev_dataset, test_dataset

def build_prompt_dataset(tokenizer, prompt_dataset, target_dataset, max_length=512, max_k=2):
    print("Build prompt dataset...")
    random.seed(43)

    # Get subjects indices
    prompt_subjects = {}
    for i in tqdm(range(len(prompt_dataset))):
        cur_subject = format_subject(prompt_dataset[i]["subject"]).strip()
        if cur_subject not in prompt_subjects:
            prompt_subjects[cur_subject] = []
        prompt_subjects[cur_subject].append(i)
    
    print("prompt_subjects:")
    print({k: len(v) for k, v in prompt_subjects.items()})

    # Generate data
    new_data = {
        "text": [], "label": []
    }
    for line in tqdm(target_dataset):
        prompt_end = format_example(line["question"], line["choices"], line["answer"], include_answer=False)
        
        subject_name = format_subject(line["subject"]).strip()

        if subject_name not in prompt_subjects:
            continue

        indices = random.sample(prompt_subjects[subject_name], k=min(max_k, len(prompt_subjects[subject_name])))
        subject_prompt_dataset = [prompt_dataset[idx] for idx in indices]

        for k in reversed(range(0, max_k)):
            prompt = gen_prompt(subject_prompt_dataset, line["subject"], k) + prompt_end

            token_num = get_token_num(tokenizer, prompt)
            if token_num > max_length:
                # if k == 0:
                #     print(f"Token number {token_num} exceeds the model's maximum length {max_length}, although k equals to {k}.")
                continue

            new_data["text"].append(prompt)
            if isinstance(line["answer"], int):
                new_data["label"].append(OPTIONS[line["answer"]])
            elif isinstance(line["answer"], str):
                new_data["label"].append(line["answer"])
            break
    return new_data

def inference_statistics(model_wrapper, dev_dataset, module_name: str):
    print("Collecting inference statistics...")
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    model_device = next(model.parameters()).device
    
    model = inject_module(model, module_name, use_re=True)
    
    maxmin_monitor = ReduceMonitor(maxmin_fn, device=model_device)
    model.add_monitor(f"maxmin_monitor", module_name, maxmin_monitor)

    with torch.inference_mode():
        for text in tqdm(dev_dataset["text"]):
            lm_forward(text, model_wrapper=model_wrapper)

    recover_module(model)
    return maxmin_monitor

def main(out_path: str, model_path: str, datasets_path: str, dataset_path: str, dataset_sample: str,
         devsplit: str, testsplit: str, search_method:str, trans: str, trans_params: Dict[str, float]):
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    trans_list = []
    if "," in trans:
        trans_list = trans.split(",")
    else:
        trans_list.append(trans.strip())
    
    trans_list = list(set(trans_list))
    trans_list = sorted(trans_list)

    file_name_ext = f"dtcover_{search_method}"
    for trans_name in trans_list:
        file_name_ext += f"+ts-{trans_name}"
        if "Cov" in trans_name:
            file_name_ext += f"_cs-{trans_params['cover_strategy']}_sk-{trans_params['samplek']}"
        if "FGSM" in trans_name:
            file_name_ext += f"_sm-{trans_params['scale_max']:.1f}_ss-{trans_params['scale_step']:.1f}"
        if "HotFlip" in trans_name:
            file_name_ext += f"_tn-{trans_params['top_n']}"
        
        if "KMNCovMlp" in trans_name or "KMNCovPosAtten" in trans_name:
            file_name_ext += f"_a-{trans_params['alpha']:.1f}"
            file_name_ext += f"_sec-{trans_params['seck']}"
        elif "TKNCovMlp" in trans_name or "TKNCovPosAtten" in trans_name:
            file_name_ext += f"_a-{trans_params['alpha']:.1f}"
            file_name_ext += f"_tk-{trans_params['topk']}"
        elif "NCovMlp" in trans_name or "NCovPosAtten" in trans_name:
            file_name_ext += f"_a-{trans_params['alpha']:.1f}"
            file_name_ext += f"_th-{trans_params['threshold']}"
        
        if "KMNCovPosAtten" in trans_name or "KMNCovPairAtten" in trans_name:
            file_name_ext += f"_ath-{trans_params['attn_threshold']}"
        elif "TKCovPosAtten" in trans_name or "TKCovPairAtten" in trans_name:
            file_name_ext += f"_ath-{trans_params['attn_threshold']}"
        elif "NCovPosAtten" in trans_name or "NCovPairAtten" in trans_name:
            file_name_ext += f"_ath-{trans_params['attn_threshold']}"

    print(f"file name raw: {file_name_ext}")
    file_name_ext_hash = hashlib.md5(file_name_ext.encode()).hexdigest()
    print(f"file name hash: {file_name_ext_hash}")

    model_mark = model_path.split("/")[-1]
    if len(datasets_path.strip()) == 0:
        dataset_mark = dataset_path.split("/")[-1]
    else:
        dataset_mark = "mixed"
    log_file_path = os.path.join(out_path, f"{model_mark}_{dataset_mark}_{'+'.join(trans_list)}_{file_name_ext_hash}_out.txt")
    other_log_file_path = os.path.join(out_path, f"{model_mark}_{dataset_mark}_{'+'.join(trans_list)}_{file_name_ext_hash}_log.txt")
    result_file_path = os.path.join(out_path, f"{model_mark}_{dataset_mark}_{'+'.join(trans_list)}_{file_name_ext_hash}_out.csv")
    config_path = os.path.join(out_path, f"{model_mark}_{dataset_mark}_{'+'.join(trans_list)}_{file_name_ext_hash}_config.json")
    lock_path = os.path.join(out_path, f"{model_mark}_{dataset_mark}_{'+'.join(trans_list)}_{file_name_ext_hash}_lock.json")
    
    print(f"Starting: {log_file_path}")

    if os.path.exists(lock_path):
        print("The result is locked, skip the param.")
        return

    if os.path.exists(log_file_path) and os.path.exists(result_file_path) and os.path.exists(config_path):
        print("The result is existing, skip the param.")
        return
    
    # Put a lock file
    lock_dict = {
        "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(lock_dict, ensure_ascii=False, indent=4))
    
    # Write Config
    config_dict = {k:v for k, v in trans_params.items()}
    config_dict["trans"] = trans_list
    config_dict["model_path"] = model_path
    config_dict["datasets_path"] = datasets_path
    config_dict["dataset_path"] = dataset_path
    config_dict["dataset_sample"] = dataset_sample
    config_dict["devsplit"] = devsplit
    config_dict["testsplit"] = testsplit
    config_dict["search_method"] = search_method

    with open(config_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(config_dict, ensure_ascii=False, indent=4))
    
    # Write logger
    logger = get_logger("dtcover", other_log_file_path)
    
    try:
        if transformers.__version__ >= "4.37.2":
            additional_args = {"attn_implementation": "eager"}
        else:
            additional_args = {}
        if "falcon" in model_path.lower():
            additional_args["trust_remote_code"] = False
        else:
            additional_args["trust_remote_code"] = True
        model = transformers.AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto",
                                                                low_cpu_mem_usage=True, **additional_args) #, device_map="auto"
        
        tk_additional_args = {}
        if "falcon" in model_path.lower():
            tk_additional_args["trust_remote_code"] = False
        else:
            tk_additional_args["trust_remote_code"] = True
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, **tk_additional_args)
        tokenizer.model_max_length = 1024
        print(f"Tokenizer {model_path} is_fast: ", tokenizer.is_fast)
        if tokenizer._pad_token is None:
            try:
                if tokenizer.padding_side == "right":
                    tokenizer.pad_token = tokenizer.eos_token
                    print(f"Tokenizer {model_path} pad_token missing, use eos_token instead.")
                elif tokenizer.padding_side == "left":
                    tokenizer.pad_token = tokenizer.bos_token
                    print(f"Tokenizer {model_path} pad_token missing, use bos_token instead.")
            except AttributeError as e:
                print(f"Failed to set the eos_token of tokenizer {model_path}")
        model_wrapper = HuggingFaceModelForCausalLMWrapper(model, tokenizer, padding="longest")

        num_options = 4
        if "winogrande" in dataset_path.lower():
            num_options = 2
        
        # goal_function = UntargetedMultipleChoiceQuestionClassification(
        #     model_wrapper=model_wrapper,
        #     use_cache=False,
        #     model_cache_size=4,
        #     query_budget=3,
        #     num_options=num_options,
        # )
        ignore_special_tokens = 100
        model_config = model_wrapper.model.config
        if model_config.model_type == "falcon":
            ignore_special_tokens = 10
        goal_function = UntargetedNextTokenClassification(
            model_wrapper=model_wrapper,
            use_cache=False,
            model_cache_size=4,
            query_budget=3,
            ignore_special_tokens=ignore_special_tokens,
        )
        # goal_function = MinimizeBleu(model_wrapper)

        if len(datasets_path.strip()) == 0:
            print("Loading datasets...")
            dev_dataset, test_dataset = load_mmlu_dataset(dataset_path, devsplit, testsplit)
            print("Building dev datasets...")
            dev_prompt_dataset = build_prompt_dataset(tokenizer, dev_dataset, dev_dataset, max_length=512, max_k=2)
            print("Building test datasets...")
            test_prompt_dataset = build_prompt_dataset(tokenizer, dev_dataset, test_dataset, max_length=512, max_k=2)

            dev_prompt_dataset_all = dev_prompt_dataset
            test_prompt_dataset_all = test_prompt_dataset
        else:
            all_datasets = datasets_path.split(";")
            dev_prompt_dataset_all = {"label": [], "text": []}
            test_prompt_dataset_all = {"label": [], "text": []}
            for each_dataset_str in all_datasets:
                parts = each_dataset_str.split(",")
                if len(parts) == 3:
                    each_dataset, each_devsplit, each_testsplit = parts
                else:
                    each_dataset, each_devsplit, each_testsplit = parts[0], devsplit, testsplit

                print("Loading datasets...")
                dev_dataset, test_dataset = load_mmlu_dataset(each_dataset, each_devsplit, each_testsplit)
                print("Building dev datasets...")
                dev_prompt_dataset = build_prompt_dataset(tokenizer, dev_dataset, dev_dataset, max_length=512, max_k=2)
                print("Building test datasets...")
                test_prompt_dataset = build_prompt_dataset(tokenizer, dev_dataset, test_dataset, max_length=512, max_k=2)

                dev_prompt_dataset_all["label"].extend(dev_prompt_dataset["label"])
                dev_prompt_dataset_all["text"].extend(dev_prompt_dataset["text"])
                test_prompt_dataset_all["label"].extend(test_prompt_dataset["label"])
                test_prompt_dataset_all["text"].extend(test_prompt_dataset["text"])
        
        random.seed(43)
        test_size = len(test_prompt_dataset_all["label"])
        if "." in dataset_sample:
            dataset_sample_value = float(dataset_sample)
            dataset_sample_value = int(dataset_sample_value * test_size)
        else:
            dataset_sample_value = int(dataset_sample)
        if dataset_sample_value > 0:
            indices = random.sample(list(range(test_size)), k=min(test_size, dataset_sample_value))
            test_prompt_dataset_all_new = {"label": [], "text": []}
            for index in indices:
                test_prompt_dataset_all_new["label"].append(test_prompt_dataset_all["label"][index])
                test_prompt_dataset_all_new["text"].append(test_prompt_dataset_all["text"][index])
            test_prompt_dataset_all = test_prompt_dataset_all_new
        dataset = HuggingFaceDataset(datasets.Dataset.from_dict(test_prompt_dataset_all))

        # We're going to use our Banana word swap class as the attack transformation.

        # Some coverage methods need pre inference
        model_config = model_wrapper.model.config
        if any(["KMNCovMlp" in t or "NBCovMlp" in t or "SNACovMlp" in t for t in trans_list]):
            mlp_re = get_injection_module(model_config.model_type, CovKind.Mlp)
            statistics = inference_statistics(model_wrapper, dev_prompt_dataset, mlp_re)
        elif any(["KMNCovPosAtten" in t or "NBCovPosAtten" in t or "SNACovPosAtten" in t for t in trans_list]):
            atten_re = get_injection_module(model_config.model_type, CovKind.AttenWeights)
            statistics = inference_statistics(model_wrapper, dev_prompt_dataset, atten_re)

        # build transformations
        transformation_list = []
        for trans_name in trans_list:
            if trans_name == "WordSwapBanana":
                transformation = BananaWordSwap()

            elif trans_name == "HotFlip":
                transformation = WordSwapHotFlipBased(model_wrapper, top_n=trans_params["top_n"])
            elif trans_name == "FGSM":
                transformation = WordSwapFGSMBased(model_wrapper, scale_max=trans_params["scale_max"], scale_step=trans_params["scale_step"])
            
            elif trans_name == "HotFlipNCovMlp":
                transformation = WordSwapCoverageHotFlipBased(
                    model_wrapper, samplek=trans_params["samplek"], alpha=trans_params["alpha"], top_n=trans_params["top_n"],
                        cov_kind=CovKind.Mlp, logger=logger)
            elif trans_name == "HotFlipNCovPosAtten":
                transformation = WordSwapCoverageHotFlipBased(
                    model_wrapper, samplek=trans_params["samplek"], alpha=trans_params["alpha"], top_n=trans_params["top_n"],
                        cov_kind=CovKind.AttenWeights, logger=logger)
            elif trans_name == "HotFlipNCovPairAtten":
                transformation = WordSwapCoverageHotFlipBased(
                    model_wrapper, samplek=trans_params["samplek"], alpha=trans_params["alpha"], top_n=trans_params["top_n"],
                        cov_kind=CovKind.PairAttenWeights, logger=logger)
            
            elif trans_name == "FGSMNCovMlp":
                transformation = WordSwapCoverageFGSMBased(
                    model_wrapper, samplek=trans_params["samplek"], alpha=trans_params["alpha"],
                    scale_max=trans_params["scale_max"], scale_step=trans_params["scale_step"],
                    cov_kind=CovKind.Mlp, logger=logger)
            elif trans_name == "FGSMNCovPosAtten":
                transformation = WordSwapCoverageFGSMBased(
                    model_wrapper, samplek=trans_params["samplek"], alpha=trans_params["alpha"],
                    scale_max=trans_params["scale_max"], scale_step=trans_params["scale_step"],
                    cov_kind=CovKind.AttenWeights, logger=logger)
            elif trans_name == "FGSMNCovPairAtten":
                transformation = WordSwapCoverageFGSMBased(
                    model_wrapper, samplek=trans_params["samplek"], alpha=trans_params["alpha"],
                    scale_max=trans_params["scale_max"], scale_step=trans_params["scale_step"],
                    cov_kind=CovKind.PairAttenWeights, logger=logger)

            elif trans_name == "WordSwapNeighboringCharacterSwap":
                transformation = WordSwapNeighboringCharacterSwap(random_one=True, skip_last_char=True)
            elif trans_name == "WordSwapRandomCharacterDeletion":
                transformation = WordSwapRandomCharacterDeletion(random_one=True, skip_last_char=True)
            elif trans_name == "WordSwapRandomCharacterInsertion":
                transformation = WordSwapRandomCharacterInsertion(random_one=True, skip_last_char=True)
            elif trans_name == "WordSwapQWERTY":
                transformation = WordSwapQWERTY(random_one=True, skip_last_char=True)
            else:
                raise Exception("Unsupported transformation")

            transformation_list.append(transformation)
        transformation = MultiTransformation(transformation_list, select_n=3, first_k=1)

        # build metrics
        model_device = next(model_wrapper.model.parameters()).device
        atten_idx = 1
        if model_config.model_type == "falcon":
            atten_idx = 2
        metrics = []
        for trans_name in trans_list:
            if trans_name.endswith("NCovMlp"):
                metric = NCovMlp(t=trans_params["threshold"]/100.0, device=model_device)
                related_trans = [t for t in transformation_list if hasattr(t, "cov_kind") and t.cov_kind == CovKind.Mlp]
            elif trans_name.endswith("NCovPosAtten"):
                metric = NCovPosAtten(t=trans_params["attn_threshold"]/100.0, atten_idx=atten_idx, device=model_device)
                related_trans = [t for t in transformation_list if hasattr(t, "cov_kind") and t.cov_kind == CovKind.AttenWeights]
            elif trans_name.endswith("NCovPairAtten"):
                metric = NCovPairAtten(t=trans_params["attn_threshold"]/100.0, atten_idx=atten_idx, device=model_device)
                related_trans = [t for t in transformation_list if hasattr(t, "cov_kind") and t.cov_kind == CovKind.PairAttenWeights]
            else:
                continue
                # raise Exception("Invalid cov type")
            metrics.append((metric, related_trans))

        # We'll constrain modification of already modified indices and stopwords
        constraints = [
            RepeatModification(),
            StopwordModification(),
            MultiChoiceOptionsModification(),
            MultiChoiceInstructionModification(),
            MaxNumWordsModified(max_num_words=5)
        ]
        # We'll use the Greedy search method
        if search_method == "greedy":
            search_method_param = GreedySearch()
        elif search_method == "beam":
            search_method_param = BeamSearch()
        else:
            raise Exception("Unsupported search method.")
        # Now, let's make the attack from the 4 components:
        attack = Attack(
            goal_function, constraints, transformation, search_method_param,
            transformation_cache_size=4, constraint_cache_size=4
        )

        # attack
        attack_args = AttackArgs(
            random_seed=43,
            num_examples=len(dataset),
            log_to_txt=log_file_path,
            log_to_csv=result_file_path,
            disable_stdout=True,
            shuffle=False,
        )

        attacker = LLMAttacker(attack, dataset, attack_args)

        if "Cov" in trans:
            attack_results: List[AttackResult] = attack_dataset(attacker, metrics, trans_params['cover_strategy'], logger)
        else:
            attack_results: List[AttackResult] = attacker.attack_dataset()
    except Exception as e:
        traceback.print_exc()
        # Remove lock file
        if os.path.exists(lock_path):
            os.remove(lock_path) 
        
    # Remove lock file
    if os.path.exists(lock_path):
        os.remove(lock_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LM cov')
    parser.add_argument('--out-path', type=str, default='./results', help='Results save path')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-2-7b-hf', help='Model name')
    # parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Inference device')
    parser.add_argument('--datasets', type=str, default="", help='datasets path')
    parser.add_argument('--dataset-sample', type=str, default="0", help='datasets sample')
    parser.add_argument('--dataset', type=str, default="cais/mmlu", help='dataset path')
    parser.add_argument('--devsplit', type=str, default="train", help='dataset dev split')
    parser.add_argument('--testsplit', type=str, default="validation", help='dataset test split')
    parser.add_argument('--search-method', type=str, default="greedy", help='search method')
    parser.add_argument('--trans', type=str, default="FGSMNCovMlp", help='adversarial methods')
    parser.add_argument('--cover-strategy', type=str, default="all", help='coverage strategy')
    parser.add_argument('--alpha', type=float, default=0, help='cov loss alpha')
    parser.add_argument('--samplek', type=int, default=5, help='sample k neurons for coverage')
    parser.add_argument('--threshold', type=int, default=50, help='cov threshold')
    parser.add_argument('--attn-threshold', type=int, default=30, help='attention cov threshold')
    parser.add_argument('--seck', type=int, default=10, help='cov sections')
    parser.add_argument('--topk', type=int, default=3, help='cov topk')
    parser.add_argument('--topn', type=int, default=-1, help='hotflip top n to replace')
    parser.add_argument('--scale-max', type=int, default=3, help='scale max')
    parser.add_argument('--scale-step', type=float, default=0.1, help='scale step')
    args = parser.parse_args()
    print(args)
    main(args.out_path, args.model, args.datasets, args.dataset, args.dataset_sample, args.devsplit, args.testsplit, args.search_method, args.trans, {
        "cover_strategy": args.cover_strategy,
        "alpha": args.alpha,
        "samplek": args.samplek,
        "threshold": args.threshold,
        "attn_threshold": args.attn_threshold,
        "seck": args.seck,
        "topk": args.topk,
        "top_n": args.topn,
        "scale_max": args.scale_max,
        "scale_step": args.scale_step,
    })

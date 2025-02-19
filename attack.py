import datetime
import hashlib
import argparse
import json
import os
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple
import torch
import random
import time
import transformers
import datasets

from pydantic import BaseModel

# Create the goal function using the model
from textattack.goal_functions import UntargetedClassification
from textattack.goal_functions import TextToTextGoalFunction
from textattack.goal_functions import MinimizeBleu
from nncov.attack_recipes.causal.a2t_yoo_2021 import A2TYoo2021
from nncov.attack_recipes.causal.checklist_ribeiro_2020 import CheckList2020
from nncov.attack_recipes.causal.dtcover_2024 import DTCover2024
from nncov.attack_recipes.causal.fgsm_2021 import FGSM2021
from nncov.attack_recipes.causal.hotflip import HotFlipEbrahimi2017
from nncov.attack_recipes.causal.iga_wang_2019 import IGAWang2019
from nncov.attack_recipes.causal.leap_2023 import LEAP2023
from nncov.attack_recipes.causal.naive_2024 import Naive2024
from nncov.attack_recipes.causal.pruthi2019 import Pruthi2019
from nncov.core import inject_module, recover_module
from nncov.cover_attack import attack_dataset
from nncov.coverage.coverage_data import NeuronCoverage
from nncov.coverage.coverage_infer import get_covered_neurons
from nncov.data.loader import build_prompt_dataset, load_mmlu_dataset
from nncov.data.mmlu import OPTIONS, format_example, format_subject, gen_prompt
from nncov.goal_functions.classification.next_token_classification import (
    UntargetedNextTokenClassification,
)
from nncov.goal_functions.classification.mcq import (
    UntargetedMultipleChoiceQuestionClassification,
)
from nncov.inference import lm_forward
from nncov.injection import get_injection_module
from nncov.llm_attacker import LLMAttacker


from nncov.logging import get_logger
from nncov.metrics.kmncov_mlp import KMNCovMlp
from nncov.metrics.kmncov_pos_atten import KMNCovPosAtten
from nncov.metrics.nbcov_mlp import NBCovMlp
from nncov.metrics.nbcov_pos_atten import NBCovPosAtten
from nncov.metrics.ncov_mlp import NCovMlp
from nncov.metrics.ncov_pair_atten import NCovPairAtten
from nncov.metrics.ncov_pos_atten import NCovPosAtten

from textattack.metrics.recipe import AdvancedAttackMetric

from nncov.metrics.snacov_mlp import SNACovMlp
from nncov.metrics.snacov_pos_atten import SNACovPosAtten
from nncov.metrics.utils import create_metric
from nncov.models.wrappers.huggingface_model_for_causal_lm_wraper import (
    HuggingFaceModelForCausalLMWrapper,
)

# Import the dataset
from textattack.datasets import HuggingFaceDataset


from textattack import Attack

from tqdm import tqdm  # tqdm provides us a nice progress bar.
from textattack.loggers import CSVLogger  # tracks a dataframe for us.
from textattack.attack_results import SuccessfulAttackResult, AttackResult
from textattack import Attacker
from textattack import AttackArgs
from nncov.monitors.reduce_monitor import ReduceMonitor
from nncov.monitors.reduce_ops import maxmin_fn

from nncov.metrics import CovKind, CovType, NeuralMetric


class NNCovParams(BaseModel):
    alpha: float
    threshold: int
    attn_threshold: int
    seck: int
    topk: int
    top_n: int
    scale_max: float
    scale_step: float


class NNCovConfig(BaseModel):
    recipe_name: str
    trans: List[str]
    hf_model_path: str
    datasets_path: List[str]
    dataset_sample: str
    trainsplit: str
    testsplit: str

    threshold: Optional[int] = None
    attn_threshold: Optional[int] = None
    alpha: Optional[float] = None
    seck: Optional[int] = None
    topk: Optional[int] = None
    top_n: Optional[int] = None
    scale_max: Optional[float] = None
    scale_step: Optional[float] = None


def inference_maxmin_metrics(model_wrapper, train_dataset, module_name: str):
    print("Collecting inference coverage metrics...")
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    model_device = next(model.parameters()).device

    model = inject_module(model, module_name, use_re=True)

    maxmin_monitor = ReduceMonitor(maxmin_fn, device=model_device)
    model.add_monitor(f"maxmin_monitor", module_name, maxmin_monitor)

    with torch.inference_mode():
        for text in tqdm(train_dataset["text"]):
            lm_forward(text, model_wrapper=model_wrapper)

    recover_module(model)
    return maxmin_monitor


def inference_coverage_metrics(
    model_wrapper, train_dataset, metric: NeuralMetric, module_name: str
) -> NeuronCoverage:
    print("Collecting inference coverage metric...")
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer
    model_device = next(model.parameters()).device

    # model = inject_module(model, module_name, use_re=True)

    cover_neurons_all = None
    with torch.inference_mode():
        for text in tqdm(train_dataset["text"]):
            cover_neurons = get_covered_neurons(
                model_wrapper,
                text=text,
                metric=metric,
                metric_name=type(metric).__name__,
                module_name=module_name,
            )
            if cover_neurons_all is None:
                cover_neurons_all = cover_neurons
            else:
                cover_neurons_all.get_logical_or(cover_neurons)

    if cover_neurons_all is None:
        raise ValueError("No coverage data collected")
    # recover_module(model)
    return cover_neurons_all


def main(
    out_path: str,
    model_path: str,
    datasets_path: str,
    dataset_sample: str,
    trainsplit: str,
    testsplit: str,
    recipe_name: str,
    transformation: str,
    transformation_params: NNCovParams,
):
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)

    # Prepare transformations
    transformation_list = []
    if "," in transformation:
        transformation_list = transformation.split(",")
    else:
        if len(transformation.strip()) != 0:
            transformation_list.append(transformation.strip())

    transformation_list = list(set(transformation_list))
    transformation_list = sorted(transformation_list)

    # Prepare datasets
    datasets_list = []
    if ";" in datasets_path:
        datasets_list = datasets_path.split(";")
    else:
        datasets_list.append(datasets_path.strip())
    datasets_list = list(set(datasets_list))
    datasets_list = sorted(datasets_list)

    # Write Config
    config = NNCovConfig(
        recipe_name=recipe_name,
        trans=transformation_list,
        hf_model_path=model_path,
        datasets_path=datasets_list,
        dataset_sample=dataset_sample,
        trainsplit=trainsplit,
        testsplit=testsplit,
    )

    if recipe_name == "dtcover":
        config.alpha = transformation_params.alpha
        config.threshold = transformation_params.threshold
        config.attn_threshold = transformation_params.attn_threshold
    elif recipe_name == "fgsm":
        config.scale_max = transformation_params.scale_max
        config.scale_step = transformation_params.scale_step

    config_dict = config.model_dump()

    # Experiment config hash
    exp_hash = hashlib.md5(json.dumps(config_dict).encode()).hexdigest()
    print(f"Experiment config hash: {exp_hash}")

    model_mark = model_path.split("/")[-1]
    dataset_marks = []
    for dataset_path in datasets_list:
        if "," in dataset_path:
            dataset_path = dataset_path.split(",")[0]
        dataset_marks.append(dataset_path.split("/")[-1])
    dataset_mark = "+".join(dataset_marks)

    file_name_ext = f"{model_mark}_{dataset_mark}_{recipe_name}_{'+'.join(transformation_list)}_{exp_hash}"
    print(f"file name raw: {file_name_ext}")
    dir_path = file_name_ext
    if not os.path.exists(os.path.join(out_path, dir_path)):
        os.makedirs(os.path.join(out_path, dir_path), exist_ok=True)
    log_file_path = os.path.join(out_path, dir_path, f"{file_name_ext}_out.txt")
    other_log_file_path = os.path.join(out_path, dir_path, f"{file_name_ext}_log.txt")
    result_file_path = os.path.join(out_path, dir_path, f"{file_name_ext}_out.csv")
    config_path = os.path.join(out_path, dir_path, f"{file_name_ext}_config.json")
    lock_path = os.path.join(out_path, dir_path, f"{file_name_ext}_lock.json")

    print(f"Starting: {log_file_path}")

    if os.path.exists(lock_path):
        print("The result is locked, skip the param.")
        return

    if (
        os.path.exists(log_file_path)
        and os.path.exists(result_file_path)
        and os.path.exists(config_path)
    ):
        print("The result is existing, skip the param.")
        return

    # Put a lock file
    lock_dict = {
        "start_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(lock_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(lock_dict, ensure_ascii=False, indent=4))

    # Write a config file
    with open(config_path, "w", encoding="utf-8") as f:
        f.write(json.dumps(config_dict, ensure_ascii=False, indent=4))

    # Write logger
    logger = get_logger("nncov", other_log_file_path)

    try:
        # Load model
        additional_args: Dict[str, Any] = {}
        if transformers.__version__ >= "4.37.2":
            additional_args["attn_implementation"] = "eager"

        if "falcon" in model_path.lower():
            additional_args["trust_remote_code"] = False
        else:
            additional_args["trust_remote_code"] = True
        model = transformers.AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True,
            **additional_args,
        )  # , device_map="auto"

        # Load tokenizer
        tk_additional_args: Dict[str, Any] = {}
        if "falcon" in model_path.lower():
            tk_additional_args["trust_remote_code"] = False
        else:
            tk_additional_args["trust_remote_code"] = True
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_path, **tk_additional_args
        )
        tokenizer.model_max_length = 1024
        print(f"Tokenizer {model_path} is_fast: ", tokenizer.is_fast)
        if not hasattr(tokenizer, "_pad_token") or tokenizer._pad_token is None:
            try:
                if tokenizer.padding_side == "right":
                    tokenizer.pad_token = tokenizer.eos_token
                    print(
                        f"Tokenizer {model_path} pad_token missing, use eos_token instead."
                    )
                elif tokenizer.padding_side == "left":
                    tokenizer.pad_token = tokenizer.bos_token
                    print(
                        f"Tokenizer {model_path} pad_token missing, use bos_token instead."
                    )
            except AttributeError as e:
                print(f"Failed to set the eos_token of tokenizer {model_path}")
        model_wrapper = HuggingFaceModelForCausalLMWrapper(
            model, tokenizer, padding="longest"
        )

        if len(datasets_list) == 0:
            raise ValueError("No datasets provided")
        elif len(datasets_list) == 1:
            dataset_path = datasets_list[0]
            print("Loading datasets...")
            train_dataset, test_dataset = load_mmlu_dataset(
                dataset_path, trainsplit, testsplit
            )
            print("Building train datasets...")
            train_prompt_dataset = build_prompt_dataset(
                tokenizer, train_dataset, train_dataset, max_length=512, max_k=2
            )
            print("Building test datasets...")
            test_prompt_dataset = build_prompt_dataset(
                tokenizer, train_dataset, test_dataset, max_length=512, max_k=2
            )

            train_prompt_dataset_all = train_prompt_dataset
            test_prompt_dataset_all = test_prompt_dataset
        else:
            all_datasets = datasets_path.split(";")
            train_prompt_dataset_all = {"label": [], "text": []}
            test_prompt_dataset_all = {"label": [], "text": []}
            for each_dataset_str in all_datasets:
                parts = each_dataset_str.split(",")
                if len(parts) == 3:
                    each_dataset, each_trainsplit, each_testsplit = parts
                else:
                    each_dataset, each_trainsplit, each_testsplit = (
                        parts[0],
                        trainsplit,
                        testsplit,
                    )

                print("Loading datasets...")
                train_dataset, test_dataset = load_mmlu_dataset(
                    each_dataset, each_trainsplit, each_testsplit
                )
                print("Building dev datasets...")
                train_prompt_dataset = build_prompt_dataset(
                    tokenizer, train_dataset, train_dataset, max_length=512, max_k=2
                )
                print("Building test datasets...")
                test_prompt_dataset = build_prompt_dataset(
                    tokenizer, train_dataset, test_dataset, max_length=512, max_k=2
                )

                train_prompt_dataset_all["label"].extend(train_prompt_dataset["label"])
                train_prompt_dataset_all["text"].extend(train_prompt_dataset["text"])
                test_prompt_dataset_all["label"].extend(test_prompt_dataset["label"])
                test_prompt_dataset_all["text"].extend(test_prompt_dataset["text"])

        # Sample the test dataset
        random.seed(43)
        test_size = len(test_prompt_dataset_all["label"])
        if "." in dataset_sample:
            dataset_sample_value = float(dataset_sample)
            dataset_sample_value = int(dataset_sample_value * test_size)
        else:
            dataset_sample_value = int(dataset_sample)
        if dataset_sample_value > 0:
            indices = random.sample(
                list(range(test_size)), k=min(test_size, dataset_sample_value)
            )
            test_prompt_dataset_all_new = {"label": [], "text": []}
            for index in indices:
                test_prompt_dataset_all_new["label"].append(
                    test_prompt_dataset_all["label"][index]
                )
                test_prompt_dataset_all_new["text"].append(
                    test_prompt_dataset_all["text"][index]
                )
            test_prompt_dataset_all = test_prompt_dataset_all_new
        dataset = HuggingFaceDataset(
            datasets.Dataset.from_dict(test_prompt_dataset_all)
        )

        # Inference in training dataset
        metric_coverage: List[Tuple[str, NeuronCoverage, Dict[str, Any]]] = []
        if recipe_name == "dtcover":
            model_config = model_wrapper.model.config
            model_device = str(next(model_wrapper.model.parameters()).device)
            for trans in transformation_list:
                print(
                    f"Inference in training dataset on transformation: {recipe_name}.{trans}"
                )
                if "mlp" in trans.lower():
                    mlp_re = get_injection_module(model_config.model_type, CovKind.Mlp)
                    module_name = mlp_re
                elif "atten" in trans.lower():
                    atten_re = get_injection_module(
                        model_config.model_type, CovKind.AttenWeights
                    )
                    module_name = atten_re
                else:
                    raise Exception(f"Unsupported coverage: {trans}")

                metric = create_metric(
                    metric_name=trans,
                    params=transformation_params.model_dump(),
                    model_device=model_device,
                )
                cover_neurons = inference_coverage_metrics(
                    model_wrapper=model_wrapper,
                    train_dataset=train_prompt_dataset_all,
                    metric=metric,
                    module_name=module_name,
                )
                metric_coverage.append(
                    (trans, cover_neurons, transformation_params.model_dump())
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

        if recipe_name == "naive":
            assert len(transformation_list) == 1
            attack = Naive2024.build(
                model_wrapper=model_wrapper, naive_method=transformation_list[0]
            )
        elif recipe_name == "dtcover":
            # assert len(transformation_list) == 1
            attack = DTCover2024.build(
                model_wrapper=model_wrapper,
                metrics=metric_coverage,
            )
        elif recipe_name == "hotflip":
            attack = HotFlipEbrahimi2017.build(model_wrapper=model_wrapper)
        elif recipe_name == "fgsm":
            attack = FGSM2021.build(
                model_wrapper=model_wrapper,
                scale_max=transformation_params.scale_max,
                scale_step=transformation_params.scale_step,
            )
        elif recipe_name == "leap":
            attack = LEAP2023.build(model_wrapper=model_wrapper)
        elif recipe_name == "a2t":
            attack = A2TYoo2021.build(model_wrapper=model_wrapper)
        elif recipe_name == "checklist":
            attack = CheckList2020.build(model_wrapper=model_wrapper)
        elif recipe_name == "pruthi":
            attack = Pruthi2019.build(model_wrapper=model_wrapper)
        elif recipe_name == "iga":
            attack = IGAWang2019.build(model_wrapper=model_wrapper)
        else:
            raise Exception("This recipe is not supported yet.")

        attacker = LLMAttacker(attack, dataset, attack_args)

        start_timestamp = time.time()
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if "dtcover" in recipe_name:
            attack_results: List[AttackResult] = attack_dataset(attacker, metric_coverage, "all", logger)
        else:
            attack_results: List[AttackResult] = attacker.attack_dataset()

        end_timestamp = time.time()
        end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logger.info(f"Start time: {start_time}")
        logger.info(f"End time: {end_time}")
        logger.info(f"Total time: {end_timestamp - start_timestamp}s")

    except Exception as e:
        traceback.print_exc()
        # Remove lock file
        if os.path.exists(lock_path):
            os.remove(lock_path)

    # Remove lock file
    if os.path.exists(lock_path):
        os.remove(lock_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LM cov")
    parser.add_argument(
        "--out-path", type=str, default="./results", help="Results save path"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-2-7b-chat-hf",
        help="Model name",
    )
    parser.add_argument("--device", type=str, default="cuda", help="Inference device")
    parser.add_argument(
        "--datasets", type=str, default="XiaHan19/winogrande4MC", help="datasets path"
    )  # cais/mmlu, XiaHan19/cmmlu
    parser.add_argument(
        "--dataset-sample", type=str, default="0", help="datasets sample"
    )
    parser.add_argument(
        "--trainsplit", type=str, default="train", help="dataset train split"
    )
    parser.add_argument(
        "--testsplit", type=str, default="validation", help="dataset test split"
    )
    parser.add_argument("--recipe", type=str, default="hotflip", help="attack recipe")
    parser.add_argument(
        "--transformation", type=str, default="", help="adversarial methods"
    )
    # parser.add_argument('--cover-strategy', type=str, default="all", help='coverage strategy')
    parser.add_argument("--alpha", type=float, default=0.5, help="cov loss alpha")
    # parser.add_argument('--samplek', type=int, default=5, help='sample k neurons for coverage')
    parser.add_argument("--threshold", type=int, default=50, help="cov threshold")
    parser.add_argument(
        "--attn-threshold", type=int, default=30, help="attention cov threshold"
    )
    parser.add_argument("--seck", type=int, default=10, help="cov sections")
    parser.add_argument("--topk", type=int, default=3, help="cov topk")
    parser.add_argument("--topn", type=int, default=-1, help="hotflip top n to replace")
    parser.add_argument("--scale-max", type=float, default=1.1, help="scale max")
    parser.add_argument("--scale-step", type=float, default=0.02, help="scale step")
    args = parser.parse_args()
    print(args)
    main(
        args.out_path,
        args.model,
        args.datasets,
        args.dataset_sample,
        args.trainsplit,
        args.testsplit,
        args.recipe,
        args.transformation,
        NNCovParams(
            alpha=args.alpha,
            threshold=args.threshold,
            attn_threshold=args.attn_threshold,
            seck=args.seck,
            topk=args.topk,
            top_n=args.topn,
            scale_max=args.scale_max,
            scale_step=args.scale_step,
        ),
    )

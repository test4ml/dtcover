
import random
import datasets
from joblib import Memory
from tqdm import tqdm

from nncov.data.mmlu import OPTIONS, format_example, format_subject, gen_prompt
from nncov.utils import get_token_num
location = '~/.cache/nncov_cache'
memory = Memory(location, verbose=0)

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

# @memory.cache
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

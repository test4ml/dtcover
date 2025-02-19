import datasets
import pandas as pd
from tqdm import tqdm

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

def load_mmlu_dataset(dataset_path: str, split: str):
    split_dataset = load_dataset_all(dataset_path, split)
    return split_dataset

def get_splits(dataset_path: str):
    all_dataset = datasets.load_dataset(dataset_path)
    return list(all_dataset.keys())

dataset_names = ["XiaHan19/winogrande4MC", "XiaHan19/ai2_arc4MC", "cais/mmlu"]

for dataset_path in dataset_names:
    print(dataset_path)
    if "mmlu" in dataset_path:
        splits = ['test', 'validation', 'dev']
    else:
        splits = get_splits(dataset_path)
    for split in splits:
        split_dataset = load_mmlu_dataset(dataset_path, split)
        print(f"{split}: {len(split_dataset)}")
    
    if "mmlu" in dataset_path:
        mmlu_train = datasets.load_dataset(dataset_path, "auxiliary_train", split="train", trust_remote_code=True)
        print(f"train: {len(mmlu_train)}")
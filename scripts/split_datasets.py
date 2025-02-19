import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

import datasets
from nncov.data.loader import load_dataset_all
from nncov.utils import save_to_dick

def split_single_dataset(dataset_name: str, split_name: str, split_num: int = 1000):
    print(dataset_name, split_name)
    if "," in split_name:
        splits = split_name.split(",")
        single_dataset = None
        for split in splits:
            each_dataset = load_dataset_all(dataset_name, split)
            if single_dataset is None:
                single_dataset = each_dataset
            else:
                single_dataset = datasets.concatenate_datasets([single_dataset, each_dataset])
    else:
        single_dataset = load_dataset_all(dataset_name, split_name)
    shuffled_dataset = single_dataset.shuffle(seed=42)
    sub_dataset = shuffled_dataset.select(list(range(split_num)))
    
    train_ratio = 0.6
    valid_ratio = 0.2
    
    train_split = int(train_ratio * split_num)
    valid_split = int((train_ratio + valid_ratio) * split_num)
    
    hf_dataset = datasets.DatasetDict(
        {
            "train": sub_dataset.select(list(range(train_split))),
            "valid": sub_dataset.select(list(range(train_split,valid_split))),
            "test": sub_dataset.select(list(range(valid_split, len(sub_dataset)))),
        }
    )
    return hf_dataset

def main():
    all_datasets = ["XiaHan19/winogrande4MC", "XiaHan19/ai2_arc4MC", "cais/mmlu"]
    for dataset in all_datasets:
        if dataset == "XiaHan19/winogrande4MC":
            split_name = "train"
            split_num = 5000
        elif dataset == "XiaHan19/ai2_arc4MC":
            split_name = "train,test"
            split_num = 5000
        elif dataset == "cais/mmlu":
            split_name = "test"
            split_num = 5000

        hf_dataset = split_single_dataset(dataset, split_name, split_num=split_num)
        out_name = dataset.split("/")[-1]
        save_to_dick(hf_dataset, f"datasets/{out_name}")

if __name__ == "__main__":
    main()
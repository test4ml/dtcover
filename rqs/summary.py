
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

import json
import pandas as pd
import glob
import re

from dtcover.result_parser import Result


def merge_dicts(dict1, dict2):
    merged_dict = dict1.copy()
    for key, value in dict2.items():
        if key in merged_dict:
            merged_dict[key].extend(value)
        else:
            merged_dict[key] = value
    return merged_dict

DATASETS = [
    "cais/mmlu",
    "XiaHan19/ai2_arc4MC",
    "XiaHan19/winogrande4MC",
]
DATASETS = [dataset_path.split("/")[-1] for dataset_path in DATASETS]

MODELS = [
    "meta-llama/Llama-2-7b-hf",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "internlm/internlm2-7b",
]

dataframe_all = {
    "Model": [], "Dataset": [], "Method": [], "FileName": [],
    "Success": [], "Failure": [], "Skip": [],
    "Original accuracy": [], "Accuracy under attack": [],
    "Attack success rate": [], "Average perturbed word": [],
}

for model in MODELS:
    modelname = model.split("/")[-1]
    for dataset_mark in DATASETS:
        dataframe = {
            "Model": [], "Dataset": [], "Method": [], "FileName": [],
            "Success": [], "Failure": [], "Skip": [],
            "Original accuracy": [], "Accuracy under attack": [],
            "Attack success rate": [], "Average perturbed word": [],
            "TempSortKey": [],
        }

        config_pattern = f"results/{modelname}_{dataset_mark}_*_config.json"
        configs = glob.glob(config_pattern)
        for config_path in configs:
            data = Result(config_path)
            trans_list = data.config.data["trans"]
            new_trans_list = []
            for trans in trans_list:
                new_trans = trans
                if new_trans.startswith("WordSwap"):
                    new_trans = new_trans[len("WordSwap"):]
                if new_trans.startswith("Neighboring"):
                    new_trans = new_trans[len("Neighboring"):]
                if new_trans.startswith("Random"):
                    new_trans = new_trans[len("Random"):]
                if len(trans_list) != 1 and new_trans.startswith("FGSM") and new_trans != "FGSM":
                    new_trans = new_trans[len("FGSM"):]
                if len(trans_list) != 1 and new_trans.startswith("HotFlip") and new_trans != "HotFlip":
                    new_trans = new_trans[len("HotFlip"):]
                new_trans_list.append(new_trans)
            trans_list = new_trans_list
            if "+".join(trans_list) == "FGSM+NCovMlp+NCovPairAtten+NCovPosAtten":
                trans_list = ["DTCover(FGSM-based)"]
            if "+".join(trans_list) == "HotFlip+NCovMlp+NCovPairAtten+NCovPosAtten":
                trans_list = ["DTCover(HotFlip-based)"]
            
            dataframe["Model"].append(data.config.data["model_path"].split("/")[-1])
            dataframe["Dataset"].append(data.config.data["dataset_path"])
            dataframe["Method"].append("+".join(trans_list))
            dataframe["FileName"].append(config_path)
            dataframe["Success"].append(data.result.summary["success"])
            dataframe["Failure"].append(data.result.summary["failure"])
            dataframe["Skip"].append(data.result.summary["skip"])
            dataframe["Original accuracy"].append(data.result.summary["original_accuracy"])
            dataframe["Accuracy under attack"].append(data.result.summary["accuracy_under_attack"])
            dataframe["Attack success rate"].append(data.result.summary["attack_success_rate"])
            dataframe["Average perturbed word"].append(data.result.summary["avg_perturbed_word"])

            TempSortKey = 100
            if len(trans_list) == 1:
                if "CharacterDeletion" in trans_list:
                    TempSortKey = 5
                elif "CharacterInsertion" in trans_list:
                    TempSortKey = 7
                elif "CharacterSwap" in trans_list:
                    TempSortKey = 9
                elif "QWERTY" in trans_list:
                    TempSortKey = 11
                elif "FGSM" in trans_list:
                    TempSortKey = 13
                elif "HotFlip" in trans_list:
                    TempSortKey = 15
                elif "DTCover" in trans_list[0] and "FGSM" in trans_list[0]:
                    TempSortKey = 41
                elif "DTCover" in trans_list[0] and "HotFlip" in trans_list[0]:
                    TempSortKey = 43
            elif len(trans_list) == 2:
                if "FGSM" in trans_list and "NCovMlp" in trans_list:
                    TempSortKey = 21
                if "HotFlip" in trans_list and "NCovMlp" in trans_list:
                    TempSortKey = 41
            elif len(trans_list) == 4:
                if "FGSM" in trans_list and "NCovMlp" in trans_list and "NCovPosAtten" in trans_list and "NCovPairAtten" in trans_list:
                    TempSortKey = 23
                if "HotFlip" in trans_list and "NCovMlp" in trans_list and "NCovPosAtten" in trans_list and "NCovPairAtten" in trans_list:
                    TempSortKey = 43

            dataframe["TempSortKey"].append(TempSortKey)

            temp_df = pd.DataFrame(dataframe).sort_values(["TempSortKey"])
            dataframe = temp_df.to_dict(orient="list")

        dataframe_all = merge_dicts(dataframe_all, dataframe)
df = pd.DataFrame(dataframe_all).sort_values(["Dataset", "Model"])
df['Dataset'] = df['Dataset'].replace(
    {
        'XiaHan19/ai2_arc4MC': 'ai2_arc',
        'XiaHan19/winogrande4MC': 'winogrande',
        'cais/mmlu': 'mmlu',
    }
)
df['Model'] = df['Model'].replace(
    {
        'Llama-2-7b-hf': 'Llama-2-7b',
        'Mistral-7B-Instruct-v0.2': 'Mistral-7B-Instruct',
    }
)
df.to_csv("summary.csv")

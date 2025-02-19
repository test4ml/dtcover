
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

import json
import pandas as pd
import glob
import re

from nncov.result_parser import Result


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
    "meta-llama/Llama-2-7b-chat-hf",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "internlm/internlm2-chat-7b",
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

        config_pattern = f"results/*/{modelname}_{dataset_mark}_*_config.json"
        configs = glob.glob(config_pattern)
        for config_path in configs:
            data = Result(config_path)
            recipe = data.config.data["recipe_name"]
            trans = data.config.data["trans"]
            trans = [t for t in trans if len(t) != 0]

            dataframe["Model"].append(data.config.data["hf_model_path"].split("/")[-1])
            dataframe["Dataset"].append(",".join(data.config.data["datasets_path"]))
            if len(trans) == 0:
                method_name = recipe
            else:
                method_name = recipe + "(" + "+".join(trans) + ")"
            dataframe["Method"].append(method_name)
            dataframe["FileName"].append(config_path)
            dataframe["Success"].append(data.result.summary["success"])
            dataframe["Failure"].append(data.result.summary["failure"])
            dataframe["Skip"].append(data.result.summary["skip"])
            dataframe["Original accuracy"].append(data.result.summary["original_accuracy"])
            dataframe["Accuracy under attack"].append(data.result.summary["accuracy_under_attack"])
            dataframe["Attack success rate"].append(data.result.summary["attack_success_rate"])
            dataframe["Average perturbed word"].append(data.result.summary["avg_perturbed_word"])
            
            TempSortKey = 100
            
            if "naive" in recipe:
                if "WordSwapRandomCharacterDeletion" in trans:
                    TempSortKey = 5
                elif "WordSwapRandomCharacterInsertion" in trans:
                    TempSortKey = 7
                elif "WordSwapNeighboringCharacterSwap" in trans:
                    TempSortKey = 9
                elif "WordSwapQWERTY" in trans:
                    TempSortKey = 11
            elif "fgsm" in recipe:
                TempSortKey = 13
            elif "hotflip" in recipe:
                TempSortKey = 15
            elif "leap" in recipe:
                TempSortKey = 17
            elif "a2t" in recipe:
                TempSortKey = 21
            elif "checklist" in recipe:
                TempSortKey = 23
            elif "iga" in recipe:
                TempSortKey = 25
            elif "pruthi" in recipe:
                TempSortKey = 27

            elif "dtcover" in recipe and "NCovMlp" in trans:
                TempSortKey = 41
            elif "dtcover" in recipe and "NCovPosAtten" in trans:
                TempSortKey = 43
            elif "dtcover" in recipe and "NCovPairAtten" in trans:
                TempSortKey = 45

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
df.to_csv("rq_results/summary.csv")

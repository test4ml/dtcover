
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
    "Time Start": [], "Time End": [], "Time": []
}

for model in MODELS:
    modelname = model.split("/")[-1]
    for dataset_mark in DATASETS:
        dataframe = {
            "Model": [], "Dataset": [], "Method": [], "FileName": [],
            "Time Start": [], "Time End": [], "Time": [], "TempSortKey": [],
        }

        log_pattern = f"results/*/{modelname}_{dataset_mark}_*_log.txt"
        configs = glob.glob(log_pattern)
        """
        解析如下
        [2025-01-21 18:10:17,590] [nncov INFO]: Start time: 2025-01-21 17:54:32
        [2025-01-21 18:10:17,590] [nncov INFO]: End time: 2025-01-21 18:10:17
        [2025-01-21 18:10:17,590] [nncov INFO]: Total time: 945.2768959999084s
        """
        
        log_pattern = f"results/*/{modelname}_{dataset_mark}_*_log.txt"
        log_files = glob.glob(log_pattern)

        for log_file in log_files:
            with open(log_file, 'r', encoding="utf-8") as file:
                log_content = file.read()
            config_file = log_file.replace("_log.txt", "_config.json")
            with open(config_file, 'r', encoding="utf-8") as file:
                config = json.load(file)
            recipe = config["recipe_name"]
            trans = config["trans"]

            # 解析日志文件中的时间信息
            time_start_match = re.search(r"Start time: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", log_content)
            time_end_match = re.search(r"End time: (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", log_content)
            total_time_match = re.search(r"Total time: (\d+\.\d+)s", log_content)

            if time_start_match and time_end_match and total_time_match:
                time_start = time_start_match.group(1)
                time_end = time_end_match.group(1)
                total_time = total_time_match.group(1)

                # 提取方法名
                if len(trans) == 0:
                    method_name = recipe
                else:
                    method_name = recipe + "(" + "+".join(trans) + ")"

                # 添加到dataframe中
                dataframe["Model"].append(modelname)
                dataframe["Dataset"].append(dataset_mark)
                dataframe["Method"].append(method_name)
                dataframe["FileName"].append(os.path.basename(log_file))
                dataframe["Time Start"].append(time_start)
                dataframe["Time End"].append(time_end)
                dataframe["Time"].append(total_time)
                    
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
dataframe_all_df = pd.DataFrame(dataframe_all).sort_values(["Dataset", "Model"])
# 将最终结果保存为CSV文件
dataframe_all_df.to_csv("rq_results/time_efficiency.csv", index=False)

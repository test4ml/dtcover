
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

import argparse
parser = argparse.ArgumentParser(description="discussion_lines_with_threshold")

# "NCovMlp", "NCovPairAtten", "NCovPosAtten"
parser.add_argument(
    "--metric",
    type=str,
    choices=["NCovMlp", "NCovPairAtten", "NCovPosAtten"],
    required=True,
    help="Specify the model to use. Choices: NCovMlp, NCovPairAtten, NCovPosAtten"
)

args = parser.parse_args()

print(f"Selected metric: {args.metric}")

MODELS = [
    "meta-llama/Llama-2-7b-chat-hf",
]

data_df = {
    "samples": [],
    "threshold": [],
    "coverage": [],
}

for model in MODELS:
    modelname = model.split("/")[-1]
    dataset_mark = "ai2_arc4MC+mmlu+winogrande4MC"

    config_pattern = f"results_search_grid/*/{modelname}_{dataset_mark}_*_config.json"
    configs = glob.glob(config_pattern)
    for config_path in configs:
        print(config_path)
        data = Result(config_path)
        trans_list = data.config.data["trans"]

        if trans_list != [
                "NCovMlp",
                "NCovPairAtten",
                "NCovPosAtten"
            ]:
            continue

        # if data.config.data["threshold"] != data.config.data["attn_threshold"]:
        #     continue
        
        for log in data.log.logs:
            if log.kind != "total":
                continue
            if log.cover_type == args.metric:
                cover = args.metric
            else:
                continue
            if cover in ("NCovPosAtten", "NCovPairAtten"):
                threshold_value = data.config.data["attn_threshold"] / 100.0
            else:
                threshold_value = data.config.data["threshold"] / 100.0
            data_df["samples"].append(log.data_idx)
            data_df["threshold"].append(threshold_value)
            data_df["coverage"].append(log.value)

# del data_df["NCovPairAtten"]
# del data_df["NCovPosAtten"]
df = pd.DataFrame(data_df)
# Draw
import matplotlib
import matplotlib.pyplot as plt

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
font = {'size': 14, 'family': 'Times New Roman'}
plt.rc('font', **font)

plt.figure(figsize=(8, 6))

import seaborn as sns
sns.set_style("darkgrid")
sns_plot = sns.lineplot(data=df, x="samples", y="coverage", hue="threshold", legend="full")
plt.legend(title='threshold', fontsize='10', title_fontsize='14')

# 显示图形
plt.savefig("figures/discussion_with_threshold_{args.metric}.pdf")

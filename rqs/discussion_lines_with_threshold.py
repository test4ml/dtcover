
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

MODELS = [
    "meta-llama/Llama-2-7b-hf",
]

data_df = {
    "samples": [],
    "threshold": [],
    "coverage": [],
}

for model in MODELS:
    modelname = model.split("/")[-1]
    dataset_mark = "mixed"

    config_pattern = f"results_search_grid/{modelname}_{dataset_mark}_*_config.json"
    configs = glob.glob(config_pattern)
    for config_path in configs:
        data = Result(config_path)
        trans_list = data.config.data["trans"]

        if "+".join(trans_list) != "HotFlip+HotFlipNCovMlp+HotFlipNCovPairAtten+HotFlipNCovPosAtten":
            continue

        if data.config.data["threshold"] != data.config.data["attn_threshold"]:
            continue
        
        for log in data.log.logs:
            if log.kind != "total":
                continue
            # if log.cover_type == "NCovMlp/WordSwapCoverageHotFlipBased":
            #     cover = "NCovMlp"
            # if log.cover_type == "NCovPosAtten/WordSwapCoverageHotFlipBased":
            #     cover = "NCovPosAtten"
            if log.cover_type == "NCovPairAtten/WordSwapCoverageHotFlipBased":
                cover = "NCovPairAtten"
            else:
                continue
            data_df["samples"].append(log.data_idx)
            data_df["threshold"].append(data.config.data["threshold"] / 100.0)
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

import seaborn as sns
sns.set_style("darkgrid")
sns_plot = sns.lineplot(data=df, x="samples", y="coverage", hue="threshold", legend="full")
plt.legend(title='threshold', fontsize='10', title_fontsize='14')

plt.savefig("figures/discussion_with_threshold.pdf")

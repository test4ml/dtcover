
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

DATASETS = [
    "cais/mmlu",
    # "XiaHan19/ai2_arc4MC",
    # "XiaHan19/winogrande4MC",
]
DATASETS = [dataset_path.split("/")[-1] for dataset_path in DATASETS]

MODELS = [
    "meta-llama/Llama-2-7b-hf",
]

data_df = {
    "idx": [],
    "coverage": [],
    "value": [],
}

for model in MODELS:
    modelname = model.split("/")[-1]
    for dataset_mark in DATASETS:
        config_pattern = f"results/{modelname}_{dataset_mark}_*_config.json"
        configs = glob.glob(config_pattern)
        for config_path in configs:
            data = Result(config_path)
            trans_list = data.config.data["trans"]

            if "+".join(trans_list) != "FGSM+FGSMNCovMlp+FGSMNCovPairAtten+FGSMNCovPosAtten":
                continue
            
            for log in data.log.logs:
                if log.kind != "total":
                    continue
                if log.cover_type == "NCovMlp/WordSwapCoverageFGSMBased":
                    cover = "NCovMlp"
                elif log.cover_type == "NCovPosAtten/WordSwapCoverageFGSMBased":
                    cover = "NCovPosAtten"
                elif log.cover_type == "NCovPairAtten/WordSwapCoverageFGSMBased":
                    cover = "NCovPairAtten"
                else:
                    continue
                data_df["idx"].append(log.data_idx)
                data_df["coverage"].append(cover)
                data_df["value"].append(log.value)

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
sns_plot = sns.lineplot(data=df, x="idx", y="value", hue="coverage")

plt.savefig("figures/discussion.pdf")

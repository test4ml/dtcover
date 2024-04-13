
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
from dtcover import venn

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

FGSM_List = ["FGSMNCovMlp", "FGSMNCovPosAtten", "FGSMNCovPairAtten"]
HOTFLIP_LIST = ["HotFlipNCovMlp", "HotFlipNCovPosAtten", "HotFlipNCovPairAtten"]

overlaps_data = {}

for model in MODELS:
    modelname = model.split("/")[-1]
    overlaps_data[modelname] = {}
    for dataset_mark in DATASETS:
        config_pattern = f"results/{modelname}_{dataset_mark}_*_config.json"
        configs = glob.glob(config_pattern)
        for config_path in configs:
            data = Result(config_path)
            trans_list = data.config.data["trans"]
            if len(trans_list) != 1:
                continue

            if trans_list[0] not in FGSM_List:
                continue
            
            trans = trans_list[0]
            if trans not in overlaps_data[modelname]:
                overlaps_data[modelname][trans] = []
            
            for i, result in enumerate(data.result.results):
                if result.target not in ["[[[FAILED]]]", "[[[FAILED]]]", "[[[SKIPPED]]]"]:
                    overlaps_data[modelname][trans].append(f"{dataset_mark}-{i}")

            print(trans)
            print(len(overlaps_data[modelname][trans]))
            print(len(data.result.results))

import matplotlib.pyplot as plt
fig=plt.figure(figsize=(12,4))
ax1 = fig.add_subplot(141)
ax2 = fig.add_subplot(142)
ax3 = fig.add_subplot(143)

ax_list = [ax1, ax2, ax3]

ir_names = ["(a) Llama-2-7b", "(b) Mistral-7B-Instruct", "(c) internlm2-7b"]
for model_i, model in enumerate(MODELS):
    modelname = model.split("/")[-1]
    for trans in FGSM_List:
        pass
    labels = venn.get_labels([
        overlaps_data[modelname][FGSM_List[0]],
        overlaps_data[modelname][FGSM_List[1]],
        overlaps_data[modelname][FGSM_List[2]],
    ], fill=['number'])
    ax = venn.venn3_ax(ax_list[model_i], labels, names=["NCovMlp", "PosAtten", "PairAtten"], legend=(model_i==2), fontsize=11)
    ax.set_title(ir_names[model_i], y=-0.08, fontdict={'fontsize':14})

fig.savefig(f"rq_results/venn_out.pdf")
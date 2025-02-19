
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
from nncov import venn

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

# FGSM_List = ["FGSMNCovMlp", "FGSMNCovPosAtten", "FGSMNCovPairAtten"]
# HOTFLIP_LIST = ["HotFlipNCovMlp", "HotFlipNCovPosAtten", "HotFlipNCovPairAtten"]
METHOD_List = [("dtcover", "NCovMlp"), ("dtcover", "NCovPosAtten"), ("dtcover", "NCovPairAtten")]

overlaps_data = {}

for model in MODELS:
    modelname = model.split("/")[-1]
    overlaps_data[modelname] = {}
    for dataset_mark in DATASETS:
        config_pattern = f"results/*/{modelname}_{dataset_mark}_*_config.json"
        configs = glob.glob(config_pattern)
        for config_path in configs:
            data = Result(config_path)
            trans_list = data.config.data["trans"]
            # remove empty strings
            trans_list = [s for s in trans_list if len(s.strip()) != 0]
            
            # if len(trans_list) != 1:
            #     continue

            # if trans_list[0] not in METHOD_List:
            #     continue
            if len(trans_list) == 0:
                trans = ""
            else:
                trans = trans_list[0]
            # print(overlaps_data[modelname], trans)
            if trans not in overlaps_data[modelname]:
                overlaps_data[modelname][trans] = []
            
            for i, result in enumerate(data.result.results):
                if result.target not in ["[[[FAILED]]]", "[[[FAILED]]]", "[[[SKIPPED]]]"]:
                    overlaps_data[modelname][trans].append(f"{dataset_mark}-{i}")

            # print(trans)
            print(len(overlaps_data[modelname][trans]))
            print(len(data.result.results))

import matplotlib.pyplot as plt
fig=plt.figure(figsize=(14,6))
ax1 = fig.add_subplot(141)
ax2 = fig.add_subplot(142)
ax3 = fig.add_subplot(143)

# 调整子图之间的水平间距
plt.subplots_adjust(wspace=0.5)  # wspace 控制水平间距，值越大间距越大

ax_list = [ax1, ax2, ax3]

ir_names = ["(a) Llama-2-7B", "(b) Mistral-7B", "(c) Internlm2-7B"]
for model_i, model in enumerate(MODELS):
    modelname = model.split("/")[-1]
    for recipe, trans in METHOD_List:
        pass
    # print(overlaps_data)
    labels = venn.get_labels([
        overlaps_data[modelname][METHOD_List[0][1]],
        overlaps_data[modelname][METHOD_List[1][1]],
        overlaps_data[modelname][METHOD_List[2][1]],
    ], fill=['number'])
    ax = venn.venn3_ax(ax_list[model_i], labels, names=["NCovMlp", "PosAtten", "PairAtten"], legend=(model_i==2), fontsize=11)
    ax.set_title(ir_names[model_i], y=-0.08, fontdict={'fontsize':14})

fig.savefig(f"rq_results/venn_out.pdf")
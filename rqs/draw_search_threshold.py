
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from dtcover.result_parser import Result
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

model_name = "Llama-2-7b-hf"
dataset_name_short = "mixed"
files = glob.glob(f"results_search_grid/{model_name}_{dataset_name_short}_*_config.json")


df_data = {
    "filename": [],
    "trans": [],
    "threshold": [],
    "attn_threshold": [],
    "samplek": [],
    "success": [],
    "failure": [],
    "skip": [],
    "original_accuracy": [],
    "accuracy_under_attack": [],
    "attack_success_rate": [],
}

for config_path in files:
    data = Result(config_path)
    if data.result.summary is None:
        continue
    if data.config.data["threshold"] != data.config.data["attn_threshold"]:
        continue
    if data.config.data["samplek"] != 100:
        continue
    df_data["filename"].append(config_path)
    df_data["trans"].append(",".join(data.config.data["trans"]))
    df_data["threshold"].append(data.config.data["threshold"])
    df_data["attn_threshold"].append(data.config.data["attn_threshold"])
    df_data["samplek"].append(data.config.data["samplek"])
    df_data["success"].append(data.result.summary["success"])
    df_data["failure"].append(data.result.summary["failure"])
    df_data["skip"].append(data.result.summary["skip"])
    df_data["original_accuracy"].append(data.result.summary["original_accuracy"])
    df_data["accuracy_under_attack"].append(data.result.summary["accuracy_under_attack"])
    df_data["attack_success_rate"].append(data.result.summary["attack_success_rate"])

df = pd.DataFrame(df_data)
df = df.sort_values(by="threshold")

df["threshold"] = df["threshold"] / 100
df["ASR"] = df["attack_success_rate"].astype(float) / 100

# Draw
import matplotlib
import matplotlib.pyplot as plt

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
font = {'size': 14, 'family': 'Times New Roman'}
plt.rc('font', **font)

import seaborn as sns
sns.set_style("darkgrid")
sns_plot = sns.lineplot(data=df, x="threshold", y="ASR")

plt.savefig("figures/search_threshold.pdf")

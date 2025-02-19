
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

from nncov.result_parser import Result
import glob
import re
import pandas as pd
import matplotlib.pyplot as plt

# 模型名称列表
model_names = ["Llama-2-7b-hf", "Mistral-7B-Instruct-v0.2", "internlm2-7b"]

model_name = "Llama-2-7b-chat-hf"
files = glob.glob(f"results_search_grid/{model_name}_*/*_config.json")


df_data = {
    "filename": [],
    "trans": [],
    "threshold": [],
    "attn_threshold": [],
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
    df_data["filename"].append(config_path)
    df_data["trans"].append(",".join(data.config.data["trans"]))
    df_data["threshold"].append(data.config.data["threshold"])
    df_data["attn_threshold"].append(data.config.data["attn_threshold"])
    df_data["success"].append(data.result.summary["success"])
    df_data["failure"].append(data.result.summary["failure"])
    df_data["skip"].append(data.result.summary["skip"])
    df_data["original_accuracy"].append(data.result.summary["original_accuracy"])
    df_data["accuracy_under_attack"].append(data.result.summary["accuracy_under_attack"])
    df_data["attack_success_rate"].append(data.result.summary["attack_success_rate"])

df = pd.DataFrame(df_data)
df.to_csv("search_grid.csv")


draw_data = {
    "threshold": [],
    "attn_threshold": [],
    "ASR": [],
}

scanned = set()

for index, row in df.iterrows():
    threshold = row["threshold"]
    attn_threshold = row["attn_threshold"]
    asr = row["attack_success_rate"]
    t = float(threshold) / 100.0
    at = float(attn_threshold) / 100.0
    
    if (t, at) in scanned:
        continue
    
    draw_data["threshold"].append(float(threshold) / 100.0)
    draw_data["attn_threshold"].append(float(attn_threshold) / 100.0)
    draw_data["ASR"].append(float(asr))
    scanned.add((t, at))
draw_df = pd.DataFrame(draw_data)
print(draw_df)

# draw heatmap
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['font.family'] = 'Times New Roman'

# 绘制热力图
plt.figure(figsize=(6.6, 5.0))
sns.heatmap(draw_df.pivot(index="threshold", columns="attn_threshold", values="ASR"), annot=False, linewidths=.5, cmap="YlGnBu")
plt.title("Attack Success Rate")
plt.xlabel("Attention Threshold")
plt.ylabel("Threshold")
plt.savefig("./figures/search_grid.pdf")

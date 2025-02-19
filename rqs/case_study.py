
import sys
import os

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
)

import json
import pandas as pd
import glob
import re

from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, PreTrainedModel
from nncov.result_parser import Result

MODELS = [
    "/data/wangjun/models/Llama-2-7b-hf",
    # "/data2/wangjun/models/Mistral-7B-Instruct-v0.2",
    # "/data2/wangjun/models/internlm2-7b",
]

def clean_token(text: str):
    if text.startswith("[["):
        text = text[2:]
    if text.startswith("]]"):
        text = text[:-2]
    
    parts = text.split(" ")
    return parts[0]

tokenizer = AutoTokenizer.from_pretrained("/data/wangjun/models/Llama-2-7b-hf")

outputs = []
for model in MODELS:
    modelname = model.split("/")[-1]
    dataset_mark = "cais/mmlu".split("/")[-1]

    config_pattern = f"results/{modelname}_{dataset_mark}_*_config.json"
    configs = glob.glob(config_pattern)
    for config_path in configs:
        data = Result(config_path)
        trans_list = data.config.data["trans"]

        if "+".join(trans_list) != "HotFlip+HotFlipNCovMlp+HotFlipNCovPairAtten+HotFlipNCovPosAtten":
            continue

        for result in data.result.results:
            if result.target in ["[[[FAILED]]]", "[[[FAILED]]]", "[[[SKIPPED]]]"]:
                continue
            ori = clean_token(result.ori)
            trg = clean_token(result.target)
            token_ori = tokenizer.convert_ids_to_tokens([int(ori)])[0]
            token_trg = tokenizer.convert_ids_to_tokens([int(trg)])[0]
            outputs.append((ori, trg, token_ori, token_trg))
            print(ori, trg)

outputs_counter = {}
with open("rq_results/case_study.txt", "w", encoding="utf-8") as f:
    for item in outputs:
        k = f"{item[0]}\t{item[1]}\t{item[2]}\t{item[3]}"
        if k not in outputs_counter:
            outputs_counter[k] = 0
        outputs_counter[k] += 1
    
    for k, count in outputs_counter.items():
        f.write(f"{k}\t{count}\n")

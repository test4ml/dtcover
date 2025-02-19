import json
import random
import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

def write_jsonl(data, file_path):
    """
    将多个 JSON 对象写入 JSONL 文件。

    :param data: 包含 JSON 对象的可迭代对象（如列表、生成器等）
    :type data: iterable
    :param file_path: JSONL 文件的保存路径
    :type file_path: str
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for record in data:
            # 将每个记录转换为 JSON 字符串并写入文件
            file.write(json.dumps(record, ensure_ascii=False) + '\n')

import glob
from nncov.result_parser import Result

MODELS = ["Llama-2-7b-chat-hf", "Mistral-7B-Instruct-v0.2", "internlm2-chat-7b"]
DATASETS = ["winogrande4MC", "ai2_arc4MC", "mmlu"]


for model in MODELS:
    for dataset in DATASETS:
        files = glob.glob(f"results_valid/*/{model}_{dataset}_*_config.json")
        for file in files:
            with open(file, "r", encoding="utf-8") as f:
                cfg = json.loads(f.read())
            cfg_recipe = cfg["recipe_name"]
            cfg_trans = sorted(cfg["trans"])
            cfg_model_path = cfg["hf_model_path"]
            
            results_file = file
            break
        if results_file is None:
            print(f"Result is not found. {(model, dataset)}")
            continue
        results = Result(results_file)
        
        print((model, dataset))
        out_dicts = []
        for result in results.result.results:
            messages = []
            if result.target == "[[[SKIPPED]]]":
                # 原来就错了。。。
                continue
            elif result.target == "[[[FAILED]]]":
                input_text = result.mutation_clean
                output_answer = result.ori[3]
            else:
                input_text = result.mutation_clean
                output_answer = result.target[3]
            messages.append({
                "role": "user",
                "content": input_text
            })
            messages.append({
                "role": "assistant",
                "content": output_answer
            })
            out_dicts.append({
                "messages": messages
            })
        print(len(out_dicts))
        out_path = os.path.join("retrain_dataset", f"{model}_{dataset}_dtcover.jsonl")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        write_jsonl(out_dicts, out_path)

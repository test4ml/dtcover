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

RECIPES_TRANS = [
    ("naive", ["WordSwapNeighboringCharacterSwap"]),
    ("naive", ["WordSwapRandomCharacterDeletion"]),
    ("naive", ["WordSwapRandomCharacterInsertion"]),
    ("naive", ["WordSwapQWERTY"]),
    ("fgsm", []),
    ("hotflip", []),
    ("leap", []),
    ("a2t", []),
    ("checklist", []),
    ("iga", []),
    ("pruthi", []),
    ("dtcover", ["NCovMlp", "NCovPairAtten", "NCovPosAtten"]),
]

files = glob.glob("results_valid_baselines/*/Llama-2-7b-chat-hf_mmlu_*_config.json")

for recipe, trans in RECIPES_TRANS:
    if trans is None:
        trans = []
    trans = sorted(trans)
    results_file = None
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            cfg = json.loads(f.read())
        cfg_recipe = cfg["recipe_name"]
        cfg_trans = sorted(cfg["trans"])
        cfg_model_path = cfg["hf_model_path"]
        
        if (
            "llama" in cfg_model_path
            and trans == cfg_trans
            and cfg_recipe == recipe
        ):
            results_file = file
            break
    if results_file is None:
        print(f"Result is not found. {(recipe, trans)}")
        continue
    results = Result(results_file)
    
    print((recipe, trans))
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
    out_path = os.path.join("retrain_baselines_dataset", f"{recipe}_{'+'.join(trans)}.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    write_jsonl(out_dicts, out_path)

from datetime import datetime
import os
import json
import re
from typing import List, Optional

class ResultItem(object):
    def __init__(self, id: Optional[int], ori: Optional[str]=None, target: Optional[str]=None, inputs: Optional[str]=None, mutation: Optional[str]=None) -> None:
        self.id: Optional[int] = id
        self.ori = ori
        self.target = target
        self.inputs = inputs
        self.mutation = mutation
    
    @property
    def mutation_clean(self) -> Optional[str]:
        if self.mutation is None:
            return None
        return re.sub(r'\[\[(.*?)\]\]', r'\1', self.mutation)

class ResultFile(object):
    def __init__(self, path: str) -> None:
        with open(path, "r", encoding="utf-8") as f:
            self.lines = f.readlines()
        
        self.separater = self.find_summary(self.lines)
        if self.separater != -1:
            self.last_lines = "".join(self.lines[self.separater:])
            self.summary = self.parse_summary(self.last_lines)
            self.results_lines = self.lines[:self.separater]
        else:
            self.summary = None
            self.results_lines = self.lines
        
        self.results = self.split_results(self.results_lines)

    def split_results(self, lines: List[str]) -> List[ResultItem]:
        out = []
        current_item = ResultItem(-1)

        content_start_line = 0
        i = 0
        while i < len(lines):
            line = lines[i]
            if i == len(lines) - 1 or (line.startswith("---------------------------------------------") and "Result" in line):
                if current_item.target not in ["[[[SKIPPED]]]", "[[[FAILED]]]"]:
                    split_line = content_start_line + (i - content_start_line) // 2
                    current_item.inputs = "".join(lines[content_start_line:split_line])
                    current_item.mutation = "".join(lines[split_line:i]).lstrip("\n")
                else:
                    current_item.mutation = "".join(lines[content_start_line:i])
                if current_item.id != -1:
                    out.append(current_item)
                    current_item = ResultItem(-1)

                if i < len(lines) - 1 and line.startswith("---------------------------------------------") and "Result" in line:
                    result_id = re.search(r"Result (\d+)", line)
                    if result_id is None:
                        current_item.id = None
                    else:
                        current_item.id = int(result_id.group(1))

                    predict_line = lines[i + 1]
                    ori_str, trg_str = predict_line.rstrip("\n").split(" --> ")
                    current_item.ori = ori_str
                    current_item.target = trg_str

                    content_start_line = i + 3
            
            i += 1
        return out

    def find_summary(self, lines: List[str]):
        for i, line in enumerate(lines):
            if line.startswith("Number of successful attacks:"):
                return i
        return -1

    def parse_summary(self, content: str): 
        success_match = re.search(r"Number of successful attacks: (\d+)", content)
        failure_match = re.search(r"Number of failed attacks: (\d+)", content)
        skip_match = re.search(r"Number of skipped attacks: (\d+)", content)
        original_accuracy_match = re.search(r"Original accuracy: ([\d.]+)%", content)
        accuracy_under_attack_match = re.search(r"Accuracy under attack: ([\d.]+)%", content)
        attack_success_rate_match = re.search(r"Attack success rate: ([\d.]+)%", content)
        avg_perturbed_word_match = re.search(r"Average perturbed word %: ([\d.]+)%", content)

        success = success_match.group(1) if success_match else None
        failure = failure_match.group(1) if failure_match else None
        skip = skip_match.group(1) if skip_match else None
        original_accuracy = original_accuracy_match.group(1) if original_accuracy_match else None
        accuracy_under_attack = accuracy_under_attack_match.group(1) if accuracy_under_attack_match else None
        attack_success_rate = attack_success_rate_match.group(1) if attack_success_rate_match else None
        avg_perturbed_word = avg_perturbed_word_match.group(1) if avg_perturbed_word_match else None

        return {
            "success": success,
            "failure": failure,
            "skip": skip,
            "original_accuracy": original_accuracy,
            "accuracy_under_attack": accuracy_under_attack,
            "attack_success_rate": attack_success_rate,
            "avg_perturbed_word": avg_perturbed_word,
        }


class LogItem(object):
    def __init__(self, kind: str, sum_type: str, cover_type: str, data_idx: int, value: float, raw_data: Optional[str]=None) -> None:
        self.kind = kind
        self.sum_type = sum_type
        self.cover_type = cover_type
        self.data_idx = data_idx
        self.value = value
        self.raw_data = None

class LogFile(object):
    def __init__(self, log_path: str) -> None:
        self.log_path = log_path

        with open(log_path, "r", encoding="utf-8") as f:
            self.lines = f.readlines()
        
        self.logs: List[LogItem] = []
        
        # [2024-04-01 20:06:13,419] [nncov INFO]: [original]:[sample_coverage]:[NCovPairAtten]:[0]:0.0
        for line in self.lines:
            match_obj = re.match(r"^(\[[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2},[0-9]{3}\]) (.*)$", line)
            if match_obj:
                time_part = match_obj.group(1)
                time_part = self.remove_brace(time_part)
                
                datetime_object = datetime.strptime(time_part, "%Y-%m-%d %H:%M:%S,%f")

                data_part = match_obj.group(2)
                if "{" in data_part and "}" in data_part:
                    start_pos = data_part.index("{")
                    self.logs.append(LogItem(
                        kind = "json",
                        sum_type = None,
                        cover_type = None,
                        data_idx = None,
                        value = None,
                        raw_data=json.loads(data_part[start_pos:].replace("\'", "\""))
                    ))
                    continue
                data_parts = data_part.split(":")
                if len(data_parts) != 6:
                    continue
                    # raise Exception("The number of data parts is not 6")
                self.logs.append(LogItem(
                    kind = self.remove_brace(data_parts[1]),
                    sum_type = self.remove_brace(data_parts[2]),
                    cover_type = self.remove_brace(data_parts[3]),
                    data_idx = int(self.remove_brace(data_parts[4])),
                    value = float(data_parts[5]),
                ))
                
            else:
                raise Exception(f"Cannot parse line: {line}")
    def remove_brace(self, text: str) -> str:
        text = text.strip()
        if text.startswith("["):
            text = text[1:]
        if text.endswith("]"):
            text = text[:-1]
        
        return text

class ConfigFile(object):
    def __init__(self, config_path: str) -> None:
        self.config_path = config_path
        with open(self.config_path, "r", encoding="utf-8") as f:
            self.data = json.loads(f.read())

class Result(object):
    def __init__(self, config_path: str) -> None:
        if config_path.endswith("_config.json"):
            self.file_name = config_path[:-len("_config.json")]
        else:
            raise Exception("Result config_path does not end with _config.json")
        
        self.log_path = self.file_name + "_log.txt"
        self.csv_path = self.file_name + "_out.csv"
        self.out_path = self.file_name + "_out.txt"

        self.config = ConfigFile(config_path)
        self.result = ResultFile(self.out_path)
        self.log = LogFile(self.log_path)
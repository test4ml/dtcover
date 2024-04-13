
from typing import List, Union


OPTIONS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K"]

def format_subject(subject: str):
    l = subject.split("_")
    s = ""
    for entry in l:
        s += " " + entry
    return s

def format_example(question: str, options: List[str], answer: Union[int, str], include_answer=True):
    prompt = question
    k = len(options)
    for j in range(k):
        prompt += "\n{}. {}".format(OPTIONS[j], options[j])
    prompt += "\nAnswer:"
    if include_answer:
        if isinstance(answer, int):
            prompt += " {}\n\n".format(OPTIONS[answer])
        elif isinstance(answer, str):
            prompt += " {}\n\n".format(answer)
    else:
        prompt += " "
    return prompt

def gen_prompt(train_df, subject, k=-1):
    subject_name = format_subject(subject).strip()
    prompt = "The following are multiple choice questions (with answers) about {}.\n\n".format(subject_name)
    if k == -1:
        k = len(train_df)
    
    example_num = 0
    for i in range(len(train_df)):
        if example_num >= k:
            break
        cur_subject = format_subject(train_df[i]["subject"]).strip()
        if subject_name == cur_subject or len(cur_subject) == 0:
            prompt += format_example(train_df[i]["question"], train_df[i]["choices"], train_df[i]["answer"], True)
            example_num += 1
    return prompt

import json
from typing import Dict, List, Optional, Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
class ModelPredictor(object):
    def __init__(self, model_path: str, device: str="cuda", torch_dtype="auto", load_8bit=False, load_4bit=False) -> None:
        # 加载模型
        self.model_path = model_path
        args = {}
        if load_8bit:
            args["load_in_8bit"] = True
        if load_4bit:
            args["load_in_4bit"] = True

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            # attn_implementation="flash_attention_2",
            **args
        )
        self.device = device
        if not load_8bit and not load_4bit:
            self.model.to(self.device)
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # self.tokenizer.padding_side = 'left'
        # self.tokenizer.pad_token = self.tokenizer.unk_token

    def token_num(self, text: Union[str, Dict[str, str]]):
        if not isinstance(text, str):
            text = self.tokenizer.apply_chat_template(
                text,
                tokenize=False,
                add_generation_prompt=True
            )
        model_inputs = self.tokenizer([text], return_tensors="pt")
        return model_inputs.input_ids.shape[1]

    def inference_single(
        self,
        input: str,
        max_length: int = 2048,
        truncation: bool = False,
        max_new_tokens: int = 2048,
        do_sample: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        input_ids = self.tokenizer(
            [input],
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=truncation,
        )["input_ids"].to(self.device)

        gen_params = {}
        if do_sample is not None:
            gen_params["do_sample"] = do_sample
        if num_beams is not None:
            gen_params["num_beams"] = num_beams
        if temperature is not None:
            gen_params["temperature"] = temperature
        if top_k is not None:
            gen_params["top_k"] = top_k
        if top_p is not None:
            gen_params["top_p"] = top_p

        outputs_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            **gen_params
        )
        outputs = self.tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
        output_text = outputs[0]
        return output_text

    def inference_batch(
        self,
        inputs: List[str],
        max_length: int = 2048,
        truncation: bool = False,
        max_new_tokens: int = 2048,
        do_sample: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        input_ids = self.tokenizer(
            inputs,
            return_tensors="pt",
            padding="longest",
            max_length=max_length,
            truncation=truncation,
        )["input_ids"].to(self.device)

        gen_params = {}
        if do_sample is not None:
            gen_params["do_sample"] = do_sample
        if num_beams is not None:
            gen_params["num_beams"] = num_beams
        if temperature is not None:
            gen_params["temperature"] = temperature
        if top_k is not None:
            gen_params["top_k"] = top_k
        if top_p is not None:
            gen_params["top_p"] = top_p

        outputs_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            **gen_params
        )
        outputs = self.tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
        return outputs

    def chat(
        self,
        messages: Dict[str, str],
        max_new_tokens: int = 2048,
        do_sample: Optional[bool] = None,
        num_beams: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
    ):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)
        attention_mask = torch.ones(
            model_inputs.input_ids.shape, dtype=torch.long, device=self.device
        )

        gen_params = {}
        if do_sample is not None:
            gen_params["do_sample"] = do_sample
        if num_beams is not None:
            gen_params["num_beams"] = num_beams
        if temperature is not None:
            gen_params["temperature"] = temperature
        if top_k is not None:
            gen_params["top_k"] = top_k
        if top_p is not None:
            gen_params["top_p"] = top_p
        
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **gen_params
        )

        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[
            0
        ]
        return response


if __name__ == "__main__":
    predictor = ModelPredictor('../models/Qwen1.5-110B-Chat/', load_8bit=True)
    prompt = "Give me a short introduction to large language model."
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ]

    out = predictor.chat(messages)
    print(out)

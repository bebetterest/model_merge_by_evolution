import argparse
from typing import Union
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .cmmlu.hf_causal_model import eval
from .cmmlu.mp_utils import run_eval


def get_cmmlu_acc(
    output_dir: str,
    model_path: str = None,
    model: Union[AutoModelForCausalLM, None] = None,
    tokenizer: Union[AutoTokenizer, None] = None,
    shot_num: int = 5,
    data_dir: str = "data/cmmlu_data",
    inf_rate: float = 1.0,
):
    args_tmp = argparse.Namespace(
        model_name_or_path=model_path,
        lora_weights="",
        data_dir=data_dir,
        save_dir=output_dir,
        num_few_shot=shot_num,
        max_length=2048,
        load_in_8bit=False,
        with_conf=False,
        cot=False,
        inf_rate=inf_rate,
    )

    if model_path is not None:
        assert model is None and tokenizer is None
        # tokenizer = AutoTokenizer.from_pretrained(
        #     model_path,
        #     trust_remote_code=True
        # )
        # model = AutoModelForCausalLM.from_pretrained(
        #     model_path,
        #     trust_remote_code=True,
        #     device_map="auto"
        # )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            device_map="auto",
            torch_dtype=(
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float32
            ),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
        )
    else:
        assert model is not None and tokenizer is not None

    res_dict, avg_all_acc = run_eval(
        model=model,
        tokenizer=tokenizer,
        eval=eval,
        args=args_tmp,
    )
    return res_dict, avg_all_acc

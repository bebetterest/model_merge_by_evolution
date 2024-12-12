# baichuan2_7b
'''
model.embed_tokens.weight torch.Size([125696, 4096])
model.layers.0.self_attn.W_pack.weight torch.Size([12288, 4096])
model.layers.0.self_attn.o_proj.weight torch.Size([4096, 4096])
model.layers.0.mlp.gate_proj.weight torch.Size([11008, 4096])
model.layers.0.mlp.down_proj.weight torch.Size([4096, 11008])
model.layers.0.mlp.up_proj.weight torch.Size([11008, 4096])
model.layers.0.input_layernorm.weight torch.Size([4096])
model.layers.0.post_attention_layernorm.weight torch.Size([4096])
...
model.layers.31.self_attn.W_pack.weight torch.Size([12288, 4096])
model.layers.31.self_attn.o_proj.weight torch.Size([4096, 4096])
model.layers.31.mlp.gate_proj.weight torch.Size([11008, 4096])
model.layers.31.mlp.down_proj.weight torch.Size([4096, 11008])
model.layers.31.mlp.up_proj.weight torch.Size([11008, 4096])
model.layers.31.input_layernorm.weight torch.Size([4096])
model.layers.31.post_attention_layernorm.weight torch.Size([4096])
model.norm.weight torch.Size([4096])
lm_head.weight torch.Size([125696, 4096])
'''

import os
import re
import json
from typing import Dict, List

from transformers import AutoModelForCausalLM


def get_inner_global_map(
    param_list: list
) -> Dict[str, List[str]]:
    res = {
        param["name"]: ["uni-_-w"]
        for param in param_list
    }
    return res


def get_inner_layerGrouping_map(
    param_list: list
) -> Dict[str, List[str]]:
    re_template = r"model\.layers\.(\d+)\."
    res = {
        param["name"]: [
            f"layer_{re.search(re_template, param['name']).group(1)}-_-w"
        ] if re.search(re_template, param["name"]) else [f"{param['name']}-_-w"] 
        for param in param_list
    }
    return res


def get_inner_individual_map(
    param_list: list
) -> Dict[str, List[str]]:
    res = {
        param["name"]: [
            f"{param['name']}-_-w"
        ] for param in param_list
    }
    return res


def get_inner_moduleGrouping_map(
    param_list: list
) -> Dict[str, List[str]]:
    re_template = r"model\.layers\.\d+\."
    res = {
        param["name"]: [
            f"{param['name']}-_-{_['name']}-_-w"  # add param['name']} to avoid sharing with different targets but same source
            for _ in param_list
            if re.sub(re_template, "", _["name"]) == re.sub(re_template, "", param["name"])
        ]
        for param in param_list
    }
    return res


def get_inner_sizeGrouping_map(
    param_list: list
) -> Dict[str, List[str]]:
    res = {
        param["name"]: [
            f"{param['name']}-_-{_['name']}-_-w"
            for _ in param_list
            if _["shape"] == param["shape"]
        ]
        for param in param_list
    }
    return res


def get_inner_map(
    model_name: str,
    madel_path: str,
    save_dir: str = "./inner_map",
) -> dict:
    model = AutoModelForCausalLM.from_pretrained(
        madel_path,
        device_map="auto",
        trust_remote_code=True
    )

    param_list = []
    for name, param in model.named_parameters():
        param_list.append({
            "name": str(name),
            "shape": param.shape
        })

    save_path = os.path.join(save_dir, model_name)
    os.makedirs(save_path, exist_ok=True)

    inner_global_map = get_inner_global_map(param_list)
    inner_layerGrouping_map = get_inner_layerGrouping_map(param_list)
    inner_individual_map = get_inner_individual_map(param_list)
    inner_moduleGrouping_map = get_inner_moduleGrouping_map(param_list)
    inner_sizeGrouping_map = get_inner_sizeGrouping_map(param_list)

    with open(os.path.join(save_path, "inner_global_map.json"), "w",\
              encoding="utf-8") as f:
        json.dump(inner_global_map, f, ensure_ascii=False, indent=4)
    with open(os.path.join(save_path, "inner_layerGrouping_map.json"), "w",\
                encoding="utf-8") as f:
        json.dump(inner_layerGrouping_map, f, ensure_ascii=False, indent=4)
    with open(os.path.join(save_path, "inner_individual_map.json"), "w",\
              encoding="utf-8") as f:
        json.dump(inner_individual_map, f, ensure_ascii=False, indent=4)
    with open(os.path.join(save_path, "inner_moduleGrouping_map.json"), "w",\
              encoding="utf-8") as f:
        json.dump(inner_moduleGrouping_map, f, ensure_ascii=False, indent=4)
    with open(os.path.join(save_path, "inner_sizeGrouping_map.json"), "w",\
              encoding="utf-8") as f:
        json.dump(inner_sizeGrouping_map, f, ensure_ascii=False, indent=4)

    print(f"inner map saved in {save_path}!")
    return {
        "inner_global_map": inner_global_map,
        "inner_individual_map": inner_individual_map,
        "inner_moduleGrouping_map": inner_moduleGrouping_map,
        "inner_sizeGrouping_map": inner_sizeGrouping_map,
    }


if __name__ == "__main__":
    get_inner_map(
        model_name="baichuan2_7b",
        madel_path="checkpoints/baichuan2_7b_checkpoints/train_00220B",
    )

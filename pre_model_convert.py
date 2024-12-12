import os
import torch
import time
from safetensors.torch import save_model, load_model
from transformers import AutoModelForCausalLM

DIR_PATH = "checkpoints/baichuan2_7b_checkpoints"

def hfmodel_torch_2_safetensors(model_dir: str):
    tmp_time = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=(
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float32
        ),
    )
    print("loaded: ", time.time() - tmp_time)

    tmp_time = time.time()
    save_model(
        model=model,
        filename=os.path.join(model_dir, "model.safetensors"),
    )
    print("safetensors save time:", time.time() - tmp_time)

    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    for root, dirs, files in os.walk(DIR_PATH):
        for dir in dirs:
            model_dir = os.path.join(root, dir)
            print(model_dir)
            hfmodel_torch_2_safetensors(model_dir=model_dir)

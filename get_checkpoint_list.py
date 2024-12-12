import os
import json


def get_checkpoint_list(
    model_name: str,
    model_dir: str,  
    save_dir: str = "./checkpoint_list",
) -> list:
    checkpoint_list = []
    for root, dirs, files in os.walk(model_dir):
        for dir in sorted(dirs):
            checkpoint_list.append(os.path.join(model_dir, dir))

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{model_name}.json")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(checkpoint_list, f, ensure_ascii=False, indent=4)
    print(f"checkpoint list saved in {save_path}!")

    return checkpoint_list


if __name__ == "__main__":
    model_name = "baichuan2_7b"
    model_dir = "checkpoints/baichuan2_7b_checkpoints"
    get_checkpoint_list(
        model_name=model_name,
        model_dir=model_dir,
    )

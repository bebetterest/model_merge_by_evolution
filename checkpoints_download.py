from huggingface_hub import snapshot_download
import os

check_point_list = [
    "train_00220B",
    "train_00440B",
    "train_00660B",
    "train_00880B",
    "train_01100B",
    "train_01320B",
    "train_01540B",
    "train_01760B",
    "train_01980B",
    "train_02200B",
    "train_02420B",
]

dir_path = "checkpoints/baichuan2_7b_checkpoints"
os.makedirs(dir_path, exist_ok=True)

for check_point in check_point_list:
    print(f"Downloading {check_point}")
    sub_dir_path = f"{dir_path}/{check_point}"
    os.makedirs(sub_dir_path, exist_ok=True)
    snapshot_download(
        repo_id="baichuan-inc/Baichuan2-7B-Intermediate-Checkpoints",
        revision=check_point,
        local_dir=sub_dir_path,
        local_dir_use_symlinks=False,
        # force_download=True
    )

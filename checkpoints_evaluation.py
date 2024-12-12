import os
import shutil
import json
from tqdm import tqdm
import torch

from eval.evaluate_mmlu import get_mmlu_acc
from eval.evaluate_zh import get_ceval_acc
from eval.evaluate_cmmlu import get_cmmlu_acc


def evaluation_mmlu(
    model_path: str,
    output_dir: str,
    inf_rate: float = 1.0,
) -> dict:
    torch.cuda.empty_cache()
    if os.path.exists(f"{output_dir}/mmlu"):
        if os.path.exists(f"{output_dir}/evaluation_mmlu.json"):
            print(f"load previous evaluation_mmlu.json from {output_dir}...")
            with open(f"{output_dir}/evaluation_mmlu.json", "r", encoding="utf-8") as f:
                return json.load(f)
        shutil.rmtree(f"{output_dir}/mmlu")
    os.makedirs(f"{output_dir}/mmlu", exist_ok=True)
    mmlu_accs, mmlu_average_acc = get_mmlu_acc(
        model_path=model_path,
        output_dir=f"{output_dir}/mmlu",
        inf_rate=inf_rate,
    )
    res = {
        "mmlu_average_acc": mmlu_average_acc,
        "mmlu_accs": mmlu_accs,
    }
    with open(f"{output_dir}/evaluation_mmlu.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    return res


def evaluation_cmmlu(
    model_path: str,
    output_dir: str,
    inf_rate: float = 1.0,
) -> dict:
    torch.cuda.empty_cache()
    if os.path.exists(f"{output_dir}/cmmlu_5_shot"):
        if os.path.exists(f"{output_dir}/evaluation_cmmlu.json"):
            print(f"load previous evaluation_cmmlu.json from {output_dir}...")
            with open(f"{output_dir}/evaluation_cmmlu.json", "r", encoding="utf-8") as f:
                return json.load(f)
        shutil.rmtree(f"{output_dir}/cmmlu_5_shot")
    os.makedirs(f"{output_dir}/cmmlu_5_shot", exist_ok=True)
    cmmlu_accs, cmmlu_average_acc = get_cmmlu_acc(
        model_path=model_path,
        output_dir=f"{output_dir}/cmmlu",  # cmmlu_5_shot in func run_eval
        inf_rate=inf_rate,
    )
    res = {
        "cmmlu_average_acc": cmmlu_average_acc,
        "cmmlu_accs": cmmlu_accs,
    }
    with open(f"{output_dir}/evaluation_cmmlu.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    return res


def evaluation_ceval(
    model_path: str,
    output_dir: str,
    inf_rate: float = 1.0,
) -> dict:
    torch.cuda.empty_cache()
    if os.path.exists(f"{output_dir}/ceval"):
        if os.path.exists(f"{output_dir}/evaluation_ceval.json"):
            print(f"load previous evaluation_ceval.json from {output_dir}...")
            with open(f"{output_dir}/evaluation_ceval.json", "r", encoding="utf-8") as f:
                return json.load(f)
        shutil.rmtree(f"{output_dir}/ceval")
    os.makedirs(f"{output_dir}/ceval", exist_ok=True)
    ceval_accs, ceval_average_acc = get_ceval_acc(
        model_path=model_path,
        output_dir=f"{output_dir}/ceval",
        inf_rate=inf_rate,
    )
    res = {
        "ceval_average_acc": ceval_average_acc,
        "ceval_accs": ceval_accs,
    }
    with open(f"{output_dir}/evaluation_ceval.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    return res


def evaluation_all(
    model_path: str,
    output_dir: str,
    inf_rate: float = 1.0,
) -> dict:
    if os.path.exists(f"{output_dir}/evaluation_all.json"):
        with open(f"{output_dir}/evaluation_all.json", "r", encoding="utf-8") as f:
            return json.load(f)
    # print("eval on mmlu")
    # mmlu_res = evaluation_mmlu(
    #     model_path=model_path,
    #     output_dir=output_dir,
    #     inf_rate=inf_rate,
    # )
    print("eval on cmmlu")
    cmmlu_res = evaluation_cmmlu(
        model_path=model_path,
        output_dir=output_dir,
        inf_rate=inf_rate,
    )
    print("eval on ceval")
    ceval_res = evaluation_ceval(
        model_path=model_path,
        output_dir=output_dir,
        inf_rate=inf_rate,
    )
    res = {
        # **mmlu_res,
        **cmmlu_res,
        **ceval_res,
    }
    with open(f"{output_dir}/evaluation_all.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=4)
    return res


if __name__ == "__main__":
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

    for chckpoint in tqdm(check_point_list):
        print(f"evaluating {chckpoint}...")
        model_path = f"checkpoints/baichuan2_7b_checkpoints/{chckpoint}"
        output_dir = f"eval_res/baichuan2_7b_checkpoints/{chckpoint}"
        # eval_res = evaluation_mmlu(
        #     model_path=model_path,
        #     output_dir=output_dir,
        #     inf_rate=0.25,
        # )
        eval_res = evaluation_ceval(
            model_path=model_path,
            output_dir=output_dir,
            inf_rate=0.25,
        )
        # eval_res = evaluation_cmmlu(
        #     model_path=model_path,
        #     output_dir=output_dir,
        #     inf_rate=0.25,
        # )
        # eval_res = evaluation_all(
        #     model_path=model_path,
        #     output_dir=output_dir,
        #     inf_rate=0.25,
        # )
        print(eval_res)

import os
import time
import random
import argparse
from functools import partial
from typing import List, Callable

import json
import pickle
import torch
import logging
import numpy as np
from tqdm import tqdm
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from deap import base, creator, tools

from torch import load as torch_load_weights
# from safetensors.torch import load_file as safetensors_load_weights

from eval.evaluate_mmlu import get_mmlu_acc
from eval.evaluate_zh import get_ceval_acc
from eval.evaluate_cmmlu import get_cmmlu_acc

import datasets
datasets.logging.set_verbosity_error()
# datasets.disable_progress_bars()
datasets.logging.disable_progress_bar()


def get_logger(
    log_path: str
) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


###---preparation---###
def get_subtask_list(
    dataset_name: str,
) -> List[str]:
    if dataset_name == "cmmlu":
        return [
            'college_engineering_hydrology',
            'chinese_literature',
            'ethnology',
            'chinese_food_culture',
            'genetics',
            'elementary_chinese',
            'college_medical_statistics',
            'college_actuarial_science',
            'high_school_geography',
            'high_school_biology',
            'elementary_commonsense',
            'marxist_theory',
            'electrical_engineering',
            'college_law',
            'world_history',
            'chinese_teacher_qualification',
            'education',
            'sociology',
            'management',
            'logical',
            'world_religions',
            'traditional_chinese_medicine',
            'journalism',
            'international_law',
            'conceptual_physics',
            'ancient_chinese',
            'legal_and_moral_basis',
            'sports_science',
            'chinese_driving_rule',
            'college_education',
            'elementary_mathematics',
            'astronomy',
            'chinese_history',
            'arts',
            'clinical_knowledge',
            'high_school_physics',
            'global_facts',
            'elementary_information_and_technology',
            'nutrition',
            'professional_law',
            'economics',
            'human_sexuality',
            'computer_science',
            'food_science',
            'chinese_civil_service_exam',
            'philosophy',
            'professional_accounting',
            'modern_chinese',
            'high_school_politics',
            'anatomy',
            'high_school_mathematics',
            'agronomy',
            'professional_psychology',
            'business_ethics',
            'professional_medicine',
            'college_mathematics',
            'marketing',
            'public_relations',
            'jurisprudence',
            'chinese_foreign_policy',
            'security_study',
            'computer_security',
            'machine_learning',
            'college_medicine',
            'high_school_chemistry',
            'virology',
            'construction_project_management',
        ]
    elif dataset_name == "ceval":
        return [
            "computer_network",
            "operating_system",
            "computer_architecture",
            "college_programming",
            "college_physics",
            "college_chemistry",
            "advanced_mathematics",
            "probability_and_statistics",
            "discrete_mathematics",
            "electrical_engineer",
            "metrology_engineer",
            "high_school_mathematics",
            "high_school_physics",
            "high_school_chemistry",
            "high_school_biology",
            "middle_school_mathematics",
            "middle_school_biology",
            "middle_school_physics",
            "middle_school_chemistry",
            "veterinary_medicine",
            "college_economics",
            "business_administration",
            "marxism",
            "mao_zedong_thought",
            "education_science",
            "teacher_qualification",
            "high_school_politics",
            "high_school_geography",
            "middle_school_politics",
            "middle_school_geography",
            "modern_chinese_history",
            "ideological_and_moral_cultivation",
            "logic",
            "law",
            "chinese_language_and_literature",
            "art_studies",
            "professional_tour_guide",
            "legal_professional",
            "high_school_chinese",
            "high_school_history",
            "middle_school_history",
            "civil_servant",
            "sports_science",
            "plant_protection",
            "basic_medicine",
            "clinical_medicine",
            "urban_and_rural_planner",
            "accountant",
            "fire_engineer",
            "environmental_impact_assessment_engineer",
            "tax_accountant",
            "physician",
        ]
    elif dataset_name == "mmlu":
        return [
            'global_facts_dev',
            'moral_scenarios_dev',
            'professional_accounting_dev',
            'virology_dev',
            'high_school_microeconomics_dev',
            'high_school_macroeconomics_dev',
            'high_school_government_and_politics_dev',
            'high_school_european_history_dev',
            'high_school_physics_dev',
            'machine_learning_dev',
            'sociology_dev',
            'logical_fallacies_dev',
            'public_relations_dev',
            'business_ethics_dev',
            'high_school_world_history_dev',
            'college_biology_dev',
            'philosophy_dev',
            'college_mathematics_dev',
            'high_school_statistics_dev',
            'nutrition_dev',
            'formal_logic_dev',
            'computer_security_dev',
            'abstract_algebra_dev',
            'security_studies_dev',
            'marketing_dev',
            'moral_disputes_dev',
            'high_school_chemistry_dev',
            'high_school_geography_dev',
            'prehistory_dev',
            'college_physics_dev',
            'high_school_psychology_dev',
            'international_law_dev',
            'clinical_knowledge_dev',
            'medical_genetics_dev',
            'elementary_mathematics_dev',
            'professional_medicine_dev',
            'human_sexuality_dev',
            'college_medicine_dev',
            'high_school_us_history_dev',
            'college_computer_science_dev',
            'high_school_mathematics_dev',
            'electrical_engineering_dev',
            'jurisprudence_dev',
            'anatomy_dev',
            'astronomy_dev',
            'miscellaneous_dev',
            'conceptual_physics_dev',
            'high_school_biology_dev',
            'professional_law_dev',
            'high_school_computer_science_dev',
            'us_foreign_policy_dev',
            'human_aging_dev',
            'management_dev',
            'professional_psychology_dev',
            'world_religions_dev',
            'college_chemistry_dev',
            'econometrics_dev'
        ]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def get_weight_map(
    inner_map: dict,
    checkpoint_list: dict,
) -> dict:
    weight_map = {
        weight_name: {
            checkpoint: [
                f"{checkpoint}-_-{_}"
                for _ in param_list
            ]
            for checkpoint in checkpoint_list
        }
        for weight_name, param_list in inner_map.items()
    }
    return weight_map


def get_evaluation_function(
    dataset_name: str,
    inf_rate: float,
) -> Callable:
    f_map = {
        "ceval": get_ceval_acc,
        "cmmlu": get_cmmlu_acc,
        "mmlu": get_mmlu_acc,
    }
    if dataset_name not in f_map.keys():
        raise ValueError(f"Unknown dataset: {dataset_name}")
    return partial(
        f_map[dataset_name],
        inf_rate=inf_rate,
    )


def get_args():
    arg = argparse.ArgumentParser()
    arg.add_argument(
        "--special_task_name",
        type=str,
        default="default_baichuan2_7b"
    )
    arg.add_argument(
        "--work_dir",
        type=str,
        default="evo_res"
    )

    arg.add_argument("--dataset", type=str, default="ceval")  # "ceval", "cmmlu", "mmlu"
    arg.add_argument("--inf_rate", type=float, default=0.25)

    arg.add_argument(
        "--main_model",
        type=str,
        default=None
    )
    arg.add_argument(
        "--inner_map_json",
        type=str,
        default="inner_map/baichuan2_7b/inner_global_map.json"
    )
    arg.add_argument(
        "--checkpoint_list_json",
        type=str,
        default="checkpoint_list/baichuan2_7b.json"
    )

    arg.add_argument("--step", type=int, default=20)
    arg.add_argument("--population", type=int, default=20)
    arg.add_argument("--CXPB", type=float, default=0.5)
    arg.add_argument("--MUTPB", type=float, default=0.5)

    arg.add_argument("--evo_checkpoint", type=str, default=None)
    arg.add_argument("--buffer", type=str, default=None)
    arg.add_argument("--seed", type=int, default=42)

    args = arg.parse_args()

    time_str = datetime.now().strftime("%Y%m%d%H%M%S")
    args.special_task_name = f"{args.special_task_name}_{time_str}"
    args.work_dir = os.path.join(args.work_dir, args.special_task_name)
    os.makedirs(args.work_dir, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.evaluation_function = get_evaluation_function(
        args.dataset,
        args.inf_rate,
    )
    args.subtask_list = get_subtask_list(args.dataset)
    with open(args.inner_map_json, "r", encoding="utf-8") as f:
        args.inner_map = json.load(f)
    with open(args.checkpoint_list_json, "r", encoding="utf-8") as f:
        args.checkpoint_list = json.load(f)
    if args.main_model is None:
        args.main_model = args.checkpoint_list[-1]
    args.weight_map = get_weight_map(
        inner_map=args.inner_map,
        checkpoint_list=args.checkpoint_list,
    )
    args.feature_list = sorted(list(set([
        ___
        for _ in args.weight_map.values() for __ in _.values() for ___ in __
    ])))
    return args


###---init_for_evo---###
def dump_evo_checkpoint(
    evo_checkpoint: dict,
    work_dir: str,
):
    os.makedirs(work_dir, exist_ok=True)
    # checkpoint_path = os.path.join(
    #     work_dir, "evo_checkpoint_step_%04d.json" % evo_checkpoint['args']['step_idx'])
    # with open(checkpoint_path, "w", encoding="utf-8") as f:
    #     json.dump(evo_checkpoint, f, ensure_ascii=False, indent=4)
    checkpoint_path = os.path.join(work_dir, "evo_checkpoint_step_%04d.pkl" % evo_checkpoint['args'].step_idx)
    with open(checkpoint_path, "wb") as f:
        pickle.dump(evo_checkpoint, f)


def load_evo_checkpoint(
    evo_checkpoint_str: str,
) -> dict:
    # with open(evo_checkpoint_str, "r", encoding="utf-8") as f:
    #     return json.load(f)
    with open(evo_checkpoint_str, "rb") as f:
        return pickle.load(f)


def parse_param_name(
    param_name: str,
) -> dict:
    tmp_list = param_name.split("-_-")
    if len(tmp_list) == 3:
        res = {
            "checkpoint": tmp_list[0],
            "target_name": None,
            "source_name": None,
            "param_name": param_name,
        }
    elif len(tmp_list) == 4:
        res = {
            "checkpoint": tmp_list[0],
            "target_name": tmp_list[1],
            "source_name": tmp_list[2],
            "param_name": param_name,
        }
    else:
        raise ValueError(f"Unknown param_name: {param_name}")
    return res


# def decorator_rescale_each_individual_feature(
#     lim_min: float = 0.0,
#     lim_max: float = 1.0,
# ):
#     def decorator(func):
#         def wrapper(*args, **kargs):
#             offspring = func(*args, **kargs)
#             for idx, child in enumerate(offspring):
#                 f_min = min(child)
#                 f_max = max(child)
#                 if f_max == f_min:
#                     x = lim_min + (lim_max - lim_min) * 0.5
#                     for i in range(len(child)):
#                         offspring[idx][i] = x
#                 else:
#                     for i in range(len(child)):
#                         offspring[idx][i] = lim_min + (lim_max - lim_min) * (child[i] - f_min) / (f_max - f_min)
#             return offspring
#         return wrapper
#     return decorator


@torch.no_grad()
def evaluation(
    individual: List,
    eval_dir: str,
    eval_func: Callable,
    weight_map: dict,
    feature_list: List,
    checkpoint_list: List,
    subtask_list: List,
    main_model: str,
) -> tuple:
    feature_map = {
        _feature_name: _feature_value
        for _feature_name, _feature_value in zip(feature_list, individual)
    }

    accumulated_weight_dict = {
        weight_name: None
        for weight_name in weight_map.keys()
    }
    for model_checkpoint in checkpoint_list:
        print(f"accumulating weights from {model_checkpoint}...")
        tmp_time = time.time()
        # tmp_state_dict = safetensors_load_weights( # 0+15
        #     os.path.join(model_checkpoint, "model.safetensors"),
        # )
        tmp_state_dict = torch_load_weights(  # 9+9
            os.path.join(model_checkpoint, "pytorch_model.bin"),
            map_location="cpu",
        )
        print(f"loading time: {time.time() - tmp_time}")
        tmp_time = time.time()
        for weight_name, checkpoint_param_dict in weight_map.items():
            param_list = checkpoint_param_dict[model_checkpoint]
            weighted_weight_list = []
            for param_name in param_list:
                parsed_param = parse_param_name(param_name)
                if parsed_param["target_name"] is None:
                    parsed_param["target_name"] = weight_name
                if parsed_param["source_name"] is None:
                    parsed_param["source_name"] = weight_name

                assert parsed_param["target_name"] == weight_name

                w = feature_map[parsed_param["param_name"]]
                weighted_weight_list.append(
                    w * tmp_state_dict[parsed_param["source_name"]]
                )
            if accumulated_weight_dict[weight_name] is None:
                accumulated_weight_dict[weight_name] = torch.sum(
                    torch.stack(weighted_weight_list, dim=0),
                    dim=0,
                )
            else:
                accumulated_weight_dict[weight_name] =\
                    accumulated_weight_dict[weight_name] + torch.sum(
                        torch.stack(weighted_weight_list, dim=0),
                        dim=0,
                    )

        print(f"accumulating time: {time.time() - tmp_time}")
        del tmp_state_dict
        # torch.cuda.empty_cache()

    merged_model = AutoModelForCausalLM.from_pretrained(
        main_model,
        state_dict=accumulated_weight_dict,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=(
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float32
        ),
    )
    del accumulated_weight_dict
    # torch.cuda.empty_cache()
    merged_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        main_model,
        trust_remote_code=True,
    )

    accs, average_acc = eval_func(
        output_dir=eval_dir,
        model=merged_model,
        tokenizer=tokenizer,
    )
    ordered_accs = [
        accs[_] for _ in subtask_list
    ]
    return tuple(ordered_accs)


def init_individual(
    feature_size: int,
):
    individual = creator.Individual(
        [random.random() for _ in range(feature_size)]
    )
    return individual


def init_for_evo(
    args: argparse.Namespace,
) -> base.Toolbox:
    task_weights = tuple([
        1.0
        for _ in range(len(args.subtask_list))
    ])
    creator.create("FitnessMultiMax", base.Fitness, weights=task_weights)
    creator.create(
        "Individual",
        list,
        fitness=creator.FitnessMultiMax,
        birth_step=-1,
        idx=-1,
    )

    toolbox = base.Toolbox()
    toolbox.register(
        "Individual",
        init_individual,
        feature_size=len(args.feature_list),
    )
    toolbox.register(
        "population",
        tools.initRepeat,
        list,
        toolbox.Individual
    )

    toolbox.register("mate", tools.cxTwoPoint)
    # toolbox.decorate("mate", decorator_rescale_each_individual_feature())
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.5, indpb=0.75)
    # toolbox.decorate("mutate", decorator_rescale_each_individual_feature())

    toolbox.register("select", tools.selRoulette)
    # toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register(
        "evaluate",
        partial(
            evaluation,
            eval_func=args.evaluation_function,
            weight_map=args.weight_map,
            feature_list=args.feature_list,
            checkpoint_list=args.checkpoint_list,
            subtask_list=args.subtask_list,
            main_model=args.main_model,
        )
    )
    return toolbox


###---evolve_pipeline---###
def produce_individuals(
    args: argparse.Namespace,
    population: List,
    toolbox: base.Toolbox,
) -> List:
    args.step_idx = args.step_idx + 1

    # crossover
    crossover_children = []
    for child1, child2 in zip(population[::2], population[1::2]):
        if random.random() < args.CXPB:
            child1_clone = toolbox.clone(child1)
            child2_clone = toolbox.clone(child2)
            toolbox.mate(child1_clone, child2_clone)

            child1_clone.birth_step = args.step_idx
            child1_clone.idx = args.ind_idx
            args.ind_idx = args.ind_idx + 1
            del child1_clone.fitness.values

            child2_clone.birth_step = args.step_idx
            child2_clone.idx = args.ind_idx
            args.ind_idx = args.ind_idx + 1
            del child2_clone.fitness.values

            crossover_children.append(child1_clone)
            crossover_children.append(child2_clone)

    # mutation
    mutation_children = []
    for mutant in population:
        if random.random() < args.MUTPB:
            mutant_clone = toolbox.clone(mutant)
            toolbox.mutate(mutant_clone)
            mutant_clone.birth_step = args.step_idx
            mutant_clone.idx = args.ind_idx
            args.ind_idx = args.ind_idx + 1
            del mutant_clone.fitness.values
            mutation_children.append(mutant_clone)

    return args, crossover_children, mutation_children


def evolve_pipeline(
    args: argparse.Namespace,
    toolbox: base.Toolbox,
    log_path: str = "log.txt",
) -> List:
    logbook = tools.Logbook()
    logbook.header = [
        "step", "fit_ave", "fit_max", "fit_min", "fit_std"
    ]
    logger = get_logger(log_path=os.path.join(args.work_dir, log_path))

    logger.info("args: %s", args)

    if not os.path.exists(args.buffer):
        args.buffer = None
    work_buffer_path = os.path.join(args.work_dir, "buffer.json")
    if args.buffer is None and os.path.exists(work_buffer_path):
        args.buffer = work_buffer_path

    if args.buffer is None:
        eval_buffer = {}
    else:
        with open(args.buffer, "r", encoding="utf-8") as f:
            eval_buffer = json.load(f)
            print("evaluation buffer loaded!")

    args.buffer = work_buffer_path

    # init_population
    print("Init population...")
    if args.evo_checkpoint is not None:
        tmp_evo_checkpoint = load_evo_checkpoint(args.evo_checkpoint)
        # args = Namespace(**tmp_evo_checkpoint["args"])
        args = tmp_evo_checkpoint["args"]
        pop = tmp_evo_checkpoint["selected_pop"]
    else:
        args.ind_idx = 0
        args.step_idx = 0
        pop = toolbox.population(n=args.population)
        print("evaluating initial population...")
        for ind in tqdm(pop):
            ind.birth_step = args.step_idx
            ind.idx = args.ind_idx
            print(f"evaluating {ind.idx}...")
            ind_s = str(ind)
            print("ind:", ind)
            if ind_s in list(eval_buffer.keys()):
                print("evaluation buffer hit!")
                ind.fitness.values = eval_buffer[ind_s]
            else:
                ind.fitness.values = toolbox.evaluate(
                    individual=ind,
                    eval_dir=os.path.join(args.work_dir, f"eval_{ind.idx}"),
                )
                eval_buffer[ind_s] = ind.fitness.values
            logger.info("%04d %s %s", ind.idx, ind.fitness.values, ind_s)
            args.ind_idx = args.ind_idx + 1
        with open(os.path.join(args.work_dir, "buffer.json"), "w", encoding=\
                  "utf-8") as f:
            json.dump(eval_buffer, f, ensure_ascii=False, indent=4)

    # evolve_loop
    for step_idx in range(args.step_idx + 1, args.step + 1):
        print(f"Step {step_idx}/{args.step}...")
        args, crossover_children, mutation_children = produce_individuals(
            args=args,
            population=pop,
            toolbox=toolbox,
        )
        print(f"{len(crossover_children)} crossover children are created")
        print(f"{len(mutation_children)} mutation children are created")
        offspring = crossover_children + mutation_children
        print("evaluating children...")
        for ind in tqdm(offspring):
            print(f"evaluating {ind.idx}...")
            ind_s = str(ind)
            print("ind:", ind)
            if ind_s in list(eval_buffer.keys()):
                print("evaluation buffer hit!")
                ind.fitness.values = eval_buffer[ind_s]
            else:
                ind.fitness.values = toolbox.evaluate(
                    individual=ind,
                    eval_dir=os.path.join(args.work_dir, f"eval_{ind.idx}"),
                )
                eval_buffer[ind_s] = ind.fitness.values
            logger.info("%04d %s %s", ind.idx, ind.fitness.values, ind_s)
        with open(os.path.join(args.work_dir, "buffer.json"), "w", encoding=\
                  "utf-8") as f:
            json.dump(eval_buffer, f, ensure_ascii=False, indent=4)

        best_ind_idx = np.argmax([
            np.mean(_.fitness.wvalues) for _ in pop
        ])
        selected_pop = toolbox.select(
            [
                pop[idx] for idx in range(len(pop)) if idx != best_ind_idx
            ] + offspring,
            k=args.population - 1
        )
        selected_pop = [pop[best_ind_idx]] + selected_pop

        fit_list = [np.mean(_.fitness.wvalues) for _ in selected_pop]
        fit_stat = {
            "fit_ave": sum(fit_list) / len(fit_list),
            "fit_max": max(fit_list),
            "fit_min": min(fit_list),
            "fit_std": np.std(fit_list),
        }
        logbook.record(
            step=step_idx,
            **fit_stat,
        )
        with open(os.path.join(args.work_dir, "log.pkl"), "wb") as f:
            pickle.dump(logbook, f)
        print(logbook)

        checkpoint_dict = {
            "population": pop,
            "crossover_children": crossover_children,
            "mutation_children": mutation_children,
            "selected_pop": selected_pop,
            "args": args,
        }
        dump_evo_checkpoint(
            evo_checkpoint=checkpoint_dict,
            work_dir=args.work_dir,
        )

        pop = selected_pop


if __name__ == "__main__":
    args = get_args()
    toolbox = init_for_evo(args=args)
    evolve_pipeline(args=args, toolbox=toolbox)

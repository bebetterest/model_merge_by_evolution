# python3 evolution.py\
#     --special_task_name default_baichuan2_7b\
#     --work_dir evo_res\
#     --dataset ceval\
#     --inf_rate 0.25\
#     --main_model checkpoints/baichuan2_7b_checkpoints/train_00220B\
#     --inner_map_json inner_map/baichuan2_7b/inner_global_map.json\
#     --checkpoint_list_json checkpoint_list/baichuan2_7b.json\
#     --step 100\
#     --population 20\
#     --CXPB 0.5\
#     --MUTPB 0.5\
#     --evo_checkpoint evo_res/evo_checkpoint.pkl\
#     --buffer buffer.json\
#     --seed 42

python3 evolution.py\
    --special_task_name default_baichuan2_7b\
    --work_dir evo_res\
    --dataset ceval\
    --inf_rate 0.000001\
    --main_model checkpoints/baichuan2_7b_checkpoints/train_00220B\
    --inner_map_json inner_map/baichuan2_7b/inner_global_map.json\
    --checkpoint_list_json checkpoint_list/baichuan2_7b.json\
    --step 100\
    --population 20\
    --CXPB 0.5\
    --MUTPB 0.5\
    --buffer buffer.json\
    --seed 42

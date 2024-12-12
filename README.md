* Theoretically, you can flexibly evolve a target model from scratch by merging some source modules/weights/matrixes as long as the sizes of source modules are matched with those of the corresponding target modules in the inner_map here.
  
### weakness
- Loading&Operating checkpoints are slow.
- Evaluation is slow.

### quickstart
```bash
# example: merge 11 baichuan2-7b checkpoints on ceval

# download checkpoints
python3 checkpoints_download.py
# get checkpoint_list
python3 get_checkpoints_list.py
# get inner_map
python3 get_inner_map.py

# evolve
python3 evolution.py
```

### customization
* edit `get_evaluation_function`, `evaluation` and `get_subtask_list` in `evolution.py` for ur own dataset and task.
* edit `get_inner_map.py` for ur own strategy of mapping from source modules to target modules. For heterogeneous sources, customize `get_weight_map` in `evolution.py` as well.
* edit `init_for_evo`, `evolve_pipeline` and `produce_individuals` in `evolution.py` for ur own evolution strategy.

import pickle
import os

def load_pkl(pkl_path):
    with open(pkl_path, "rb") as f:
        log = pickle.load(f)
    return log


def get_latest_log_dir_path(log_dir):
    log_dir_list = [
        os.path.join(log_dir, x)
        for x in os.listdir(log_dir)
    ]
    latest_log_dir = max(log_dir_list, key=os.path.getctime)
    return latest_log_dir


if __name__ == "__main__":
    # log_pkl_path = "evo_res/default_baichuan2_7b_20240605162854/log.pkl"
    log_pkl_path = f"{get_latest_log_dir_path('evo_res')}/log.pkl"
    print(log_pkl_path)
    latest_log = load_pkl(log_pkl_path)
    print(latest_log)
    print(type(latest_log))

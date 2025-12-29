import os
import yaml
import random
import numpy as np
import torch

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok = True)

def make_run_dir(runs_root: str, run_name: str) -> str:
    run_dir = os.path.join(runs_root, run_name)
    ensure_dir(run_dir)
    return run_dir

def save_config(config: dict, path: str):
    with open(path, "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


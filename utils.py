import os
import re

SAME_SEED = "same_seed"
DIFF_SEED = "different_seeds"

RUNS = "runs"
EPOCHS = "epochs"

PRINT_START = "*" * 50 + "\n" * 3
PRINT_END = PRINT_START[::-1]  # Reversed

THIS_DIR = os.path.dirname(__file__)
print(THIS_DIR)

COMPARISON_FILEPATH = os.path.join(THIS_DIR, "comparison.txt")
SCORES_FILEPATH_FMT = os.path.join(THIS_DIR, "resnet_%s.csv")
HISTORY_FILEPATH_FMT = os.path.join(THIS_DIR, "resnet_history_%s_%d.json")

SAVED_MODELS_DIR = os.path.join(THIS_DIR, "saved_models")

def get_params():
    with open("params.txt") as f:
        lines = re.split('\r|\n', f.read(-1))
        num_runs = int(lines[0])
        num_epochs = int(lines[1])

    return {RUNS: num_runs, EPOCHS: num_epochs}

def get_num_runs():
    return get_params()[RUNS]

def get_num_epochs():
    return get_params()[EPOCHS]

def seed_to_str(same_seed):
    return SAME_SEED if same_seed else DIFF_SEED

def seed_to_str_fmt(same_seed):
    return seed_to_str(same_seed) + "%d"

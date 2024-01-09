import torch
from torch.utils.data import random_split
import json

if __name__ == "__main__":
    task = "rabbits_and_chickens"

    with open(f"{task}.json", "r") as f:
        dataset = json.load(f)

    train_set, test_set = random_split(dataset=dataset, lengths=[3030, 2020], generator=torch.Generator().manual_seed(0))

    with open(f"{task}_train.json", "w") as f:
        json.dump(list(train_set), f)

    with open(f"{task}_test.json", "w") as f:
        json.dump(list(test_set), f)

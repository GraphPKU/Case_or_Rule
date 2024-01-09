import torch
from torch.utils.data import random_split
import json

if __name__ == "__main__":
    with open(f"mod_add.json", "r") as f:
        dataset = json.load(f)

    train_set, test_set = random_split(dataset=dataset, lengths=[3830*2, len(dataset) - 3830*2], generator=torch.Generator().manual_seed(0))

    with open(f"train.json", "w") as f:
        json.dump(list(train_set), f)

    with open(f"test.json", "w") as f:
        json.dump(list(test_set), f)

import torch
from torch.utils.data import random_split
import json

length = 14

with open(f"3_digits_{length}.json", "r") as f:
    dataset = json.load(f)

train_set, test_set = random_split(dataset=dataset, lengths=[11468, 2 ** 14 - 11468], generator=torch.Generator().manual_seed(0))

with open(f"3_digits_train_{length}.json", "w") as f:
    json.dump(list(train_set), f)

with open(f"3_digits_test_{length}.json", "w") as f:
    json.dump(list(test_set), f)

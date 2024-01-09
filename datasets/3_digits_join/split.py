import torch
from torch.utils.data import random_split
import json

with open("3_digits_join.json", "r") as f:
    dataset = json.load(f)

train_set, test_set = random_split(dataset=dataset, lengths=[3000, 1096], generator=torch.Generator().manual_seed(0))

with open("3_digits_train_join.json", "w") as f:
    json.dump(list(train_set), f)

with open("3_digits_test_join.json", "w") as f:
    json.dump(list(test_set), f)

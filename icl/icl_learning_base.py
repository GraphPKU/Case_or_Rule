from base_addition_icl import BaseInt, test_a_b, is_easy_add
from tqdm import tqdm
import pickle
import random

NUM_TEST = 100
max_length = 10
task = 'rf'

few_shot_digit: list[tuple[int, int]] = [(2,2), (3,3), (5,5), (2,3), (3,5)]
accu_list = []

few_shot = []
for length in few_shot_digit:
    while True:
        a = BaseInt.random_generate_length(length[0])
        b = BaseInt.random_generate_length(length[1])
        if random.random() < 0.5:
            a, b = b, a
        if not is_easy_add(a, b):
            few_shot.append((a, b))
            break
random.shuffle(few_shot)
tqdm.write(f'Few shot examples: {few_shot}\n\n')

for longer_length in tqdm(range(1, max_length+1)):
    accu = 0
    for i in tqdm(range(NUM_TEST)):
        a = BaseInt.random_generate_length(longer_length)
        shorter_length = random.randint(1, longer_length)
        b = BaseInt.random_generate_length(shorter_length)
        if random.random() < 0.5:
            a, b = b, a
        accu += test_a_b(a, b, few_shot, test_num=10, detail=False, type=task)
    accu_list.append(accu / NUM_TEST)
    tqdm.write(f"Length {longer_length} Accuracy: {accu / NUM_TEST}")

with open(f'icl_learning_{task}_base.pkl', 'wb') as f:
    pickle.dump(accu_list, f)
    
'''
The script is to create the modular_addition(mod 113) dataset.

Here's an example:
1+113=1

we tokenize input as [tokenizer(1),tokenizer(+),tokenizer(113),tokenizer(=)

we create several squares of test set (center, length), the rest is train set

* test set: [
             [a in [center_1(a) - length_1, center_1(a) + length_1],
              b in [center_1(b) - length_1, center_1(b) + length_1]],
             [a in [center_2(a) - length_2, center_2(a) + length_2],
              b in [center_2(b) - length_2, center_2(b) + length_2]]
              ...
            ]
* train set: the rest
'''

import json
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import interval
import os

def generate_dataset(a_range, b_range, p):
    idx = 0
    samples = []
    for a in a_range:
        for b in b_range:
            a = int(a)
            b = int(b)
            samples.append({
                "id": idx,
                "question": f"{a}+{b}=",
                "a": a,
                "b": b,
                "answers": [str(np.mod(a + b, p))]
                })
            idx += 1
    print(len(samples))
    return samples

def generate_test(a_range, b_range, length_range, square_num):
    generated_square = 0
    centers = []
    lengths = []
    a_start = a_range[0]
    a_end = a_range[-1]
    b_start = b_range[0]
    b_end = b_range[-1]
    a_intervals = []
    b_intervals = []
    while generated_square < square_num:
        # generate centers
        center = (random.randint(a_start, a_end),
                  random.randint(b_start, b_end))
        # for each center, generate length, make sure that center +- length is still in the range
        length = random.randint(length_range[0], length_range[-1])
        a_interval = interval.Interval(center[0] - length, center[0] + length)
        b_interval = interval.Interval(center[1] - length, center[1] + length)
        if a_interval in interval.Interval(a_start, a_end) and b_interval in interval.Interval(b_start, b_end):
            overlap = False
            # make sure the squares do not overlap
            for idx in range(generated_square):
                if a_interval.overlaps(a_intervals[idx]) and b_interval.overlaps(b_intervals[idx]):
                    overlap = True
                    break
            if not overlap:
                centers.append(center)
                lengths.append(length)
                a_intervals.append(a_interval)
                b_intervals.append(b_interval)
                generated_square += 1
    return centers, lengths

if __name__ == "__main__":
    p = 113
    number = 5
    dataset = generate_dataset(np.arange(0,113), np.arange(0,113), p)
    centers, lengths = generate_test(np.arange(0,113), np.arange(0,113), np.arange(7,15), 5)
    test_samples = []
    train_samples = []
    img = np.zeros((p, p))
    for sample in tqdm(dataset):
        test = False
        for idx in range(len(centers)):
            center = centers[idx]
            length = lengths[idx]
            if center[0] - length <= sample["a"] <= center[0] + length and center[1] - length <= sample["b"] <= center[1] + length:
                test_samples.append(sample)
                test = True
                img[sample["a"], sample["b"]] = 1
                break
        if test: pass
        else: train_samples.append(sample)

    os.mkdir(f"{number}")

    with open(f"{number}/train.json", "w") as f:
        json.dump(train_samples, f)

    with open(f"{number}/test.json", "w") as f:
        json.dump(test_samples, f)

    with open(f"{number}/test_squares.txt", "w") as f:
        f.write("centers:{}\nlengths:{}".format(centers, lengths))

    print(f"# train: {len(train_samples)}\n# test: {len(test_samples)}\n# sum: {len(train_samples)+len(test_samples)}\n# all: {len(dataset)}")

    plt.imshow(img)
    plt.savefig(f"{number}/data_split")
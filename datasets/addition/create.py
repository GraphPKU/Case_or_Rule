'''
The script is to create the addition dataset.

Here's an example:
1+1=2
'''

import json
import math
import numpy as np
import torch
from torch.utils.data import random_split
import os
import random
import interval
from tqdm import tqdm
import matplotlib.pyplot as plt

def generate_dataset(a_range, b_range):
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
                "answers": [str(a + b)]
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
    test_num = 0
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
                test_num += (2 * length + 1) ** 2
    return centers, lengths, test_num


if __name__ == "__main__":
    start = 0
    end = 100
    dataset = generate_dataset(np.arange(start,end), np.arange(start,end))
    status = "random_split" # choose from "random_split" or "one_hole" or "multi_holes" or "column"
    status = "one_hole"
    train_ratio = 0.7
    hole_num = 3
    center_a = round((start + end)/2)
    center_b = round((start + end)/2)
    length = 8
    column_idx = random.sample(range(start, end), 21)
    row_range = [40, 60]

    if status == "random_split":
        train_num = int(len(dataset) * train_ratio)
        test_num = len(dataset) - train_num
        train_set, test_set = random_split(dataset=dataset, lengths=[train_num, test_num], generator=torch.Generator().manual_seed(42))
        title = f"random_split_{train_ratio}_{train_num}_{test_num}_{start}-{end}"
        os.mkdir(f"{title}")
        with open(f"{title}/train.json", "w") as f:
            json.dump(list(train_set), f)
        with open(f"{title}/test.json", "w") as f:
            json.dump(list(test_set), f)

    elif status == "one_hole":
        center = (center_a,center_b)
        assert length * 2 + 1 <= np.sqrt(len(dataset) * (1-train_ratio))

        title = f"1hole_{center}_{length}_{(length * 2 + 1)**2}_{start}-{end}"
        test_set = []
        train_set = []
        img = np.zeros((end-start, end-start))
        for sample in tqdm(dataset):
            test = False
            if center[0] - length <= sample["a"] <= center[0] + length and center[1] - length <= sample["b"] <= center[1] + length:
                test_set.append(sample)
                test = True
                img[sample["a"], sample["b"]] = 2
            else:
                train_set.append(sample)
                img[sample["a"], sample["b"]] = 1
        os.mkdir(f"{title}")
        with open(f"{title}/train.json", "w") as f:
            json.dump(list(train_set), f)
        with open(f"{title}/test.json", "w") as f:
            json.dump(list(test_set), f)
        plt.imshow(img)
        plt.savefig(f"{title}/data_split")

    elif status == "multi_holes":
        centers, lengths, test_num = generate_test(np.arange(start, end), np.arange(start, end), np.arange(10,20), hole_num)
        title = f"{hole_num}hole_{test_num}_{start}-{end}"
        test_set = []
        train_set = []
        img = np.zeros((end-start, end-start))
        for sample in tqdm(dataset):
            test = False
            for idx, center in enumerate(centers):
                length = lengths[idx]
                if center[0] - length <= sample["a"] <= center[0] + length and center[1] - length <= sample["b"] <= center[1] + length:
                    test_set.append(sample)
                    test = True
                    img[sample["a"], sample["b"]] = 1
                    break
            if test:pass
            else:
                train_set.append(sample)
                img[sample["a"], sample["b"]] = 0
        os.mkdir(f"{title}")
        with open(f"{title}/train.json", "w") as f:
            json.dump(list(train_set), f)
        with open(f"{title}/test.json", "w") as f:
            json.dump(list(test_set), f)
        with open(f"{title}/test_squares.txt", "w") as f:
            f.write("centers:{}\nlengths:{}".format(centers, lengths))
        plt.imshow(img)
        plt.savefig(f"{title}/data_split")

    elif status == "column":
        title = f"column_{len(column_idx)}_{row_range}_{int(len(column_idx) * (1 + row_range[1]-row_range[0]))}"

        test_set = []
        train_set = []
        img = np.zeros((end-start, end-start))

        for sample in tqdm(dataset):
            test = False
            if row_range[0] <= sample["a"] <= row_range[1] and sample["b"] in column_idx:
                test_set.append(sample)
                test = True
                img[sample["a"], sample["b"]] = 2
            else:
                train_set.append(sample)
                img[sample["a"], sample["b"]] = 1
        os.mkdir(f"{title}")
        with open(f"{title}/train.json", "w") as f:
            json.dump(list(train_set), f)
        with open(f"{title}/test.json", "w") as f:
            json.dump(list(test_set), f)
        plt.imshow(img)
        plt.savefig(f"{title}/data_split")
        with open(f"{title}/columns.txt", "w") as f:
            f.write("columns:{}\nrow_range:{}".format(column_idx, row_range))

    elif status == "row":
        title = f"row_{len(column_idx)}_{row_range}_{int(len(column_idx) * (1 + row_range[1]-row_range[0]))}"

        test_set = []
        train_set = []
        img = np.zeros((end-start, end-start))

        for sample in tqdm(dataset):
            test = False
            if row_range[0] <= sample["b"] <= row_range[1] and sample["a"] in column_idx:
                test_set.append(sample)
                test = True
                img[sample["a"], sample["b"]] = 2
            else:
                train_set.append(sample)
                img[sample["a"], sample["b"]] = 1
        os.mkdir(f"{title}")
        with open(f"{title}/train.json", "w") as f:
            json.dump(list(train_set), f)
        with open(f"{title}/test.json", "w") as f:
            json.dump(list(test_set), f)
        plt.imshow(img)
        plt.savefig(f"{title}/data_split")
        with open(f"{title}/rows.txt", "w") as f:
            f.write("columns:{}\nrow_range:{}".format(column_idx, row_range))

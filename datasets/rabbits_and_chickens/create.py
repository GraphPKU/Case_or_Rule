'''
The script is to create the rabbits_and_chickens dataset.

Here's an example:
Q: Rabbits have 4 legs and 1 head. Chickens have 2 legs and 1 head. There are <<a>> legs and <<b>> heads on the farm. How many rabbits and chickens are there?
A: There are <<(a-2b)/2>> rabbits and <<(4b-a)/2>> chickens.
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

def generate_dataset(min_num_head, max_num_head):
    samples = []
    idx = 0
    question = "Rabbits have 4 legs and 1 head. Chickens have 2 legs and 1 head. There are {} legs and {} heads on the farm. How many rabbits and chickens are there?"
    answer = "There are {} rabbits and {} chickens."
    for heads in range(min_num_head, max_num_head):
        for legs in range(2 * heads, 4 * heads + 1, 2):
            samples.append({
                "id": idx,
                "question": question.format(legs, heads),
                "answers": [answer.format(int((legs - 2 * heads)/2), int((4 * heads - legs)/2))],
                "a": legs,
                "b": heads
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
        b_center = random.randint(b_start, b_end) # head
        a_center = random.randint(b_center, b_center * 2)
        center = (a_center, b_center)
        # for each center, generate length, make sure that center +- length is still in the range
        lb = length_range[0]
        ub = min(length_range[1], int((a_center-b_center)/2), int((2*b_center-a_center)/3))
        if lb <= ub:
            length = random.randint(lb,ub)
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
                    test_num += (length * 2 + 1) ** 2
    return centers, lengths, test_num


if __name__ == "__main__":
    start = 0
    end = 100
    dataset = generate_dataset(start, end)
    # status = "random_split" # choose from "random_split" or "one_hole" or "multi_holes"
    status = "multi_holes"
    train_ratio = 0.7
    hole_num = 3
    center_head = int((start + end)/2)
    center_leg = round(center_head * 3 / 2)
    center_leg = 70
    length = 10

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
        center = (center_leg,center_head)
        assert length * 2 + 1 <= np.sqrt(len(dataset) * (1-train_ratio))

        title = f"1hole_{center}_{length}_{(length * 2 + 1)**2}_{start}-{end}"
        test_set = []
        train_set = []
        img = np.zeros((2*(end-start), end-start)) # range of (leg_num/2,head_num)
        for sample in tqdm(dataset):
            test = False
            if center[0] - length <= sample["a"]/2 <= center[0] + length and center[1] - length <= sample["b"] <= center[1] + length:
                test_set.append(sample)
                test = True
                img[round(sample["a"]/2), sample["b"]] = 2
            else:
                train_set.append(sample)
                img[round(sample["a"]/2), sample["b"]] = 1
        os.mkdir(f"{title}")
        with open(f"{title}/train.json", "w") as f:
            json.dump(list(train_set), f)
        with open(f"{title}/test.json", "w") as f:
            json.dump(list(test_set), f)
        plt.imshow(img)
        plt.savefig(f"{title}/data_split")

    elif status == "multi_holes":
        centers, lengths, test_num = generate_test(np.arange(2 * start, 2 * end), np.arange(start, end), np.arange(5,15), hole_num)
        title = f"{hole_num}hole_{test_num}_{start}-{end}"
        test_set = []
        train_set = []
        img = np.zeros((2*(end-start), end-start)) # range of (leg_num/2,head_num)
        for sample in tqdm(dataset):
            test = False
            for idx, center in enumerate(centers):
                length = lengths[idx]
                if center[0] - length <= sample["a"]/2 <= center[0] + length and center[1] - length <= sample["b"] <= center[1] + length:
                    test_set.append(sample)
                    test = True
                    img[round(sample["a"]/2), sample["b"]] = 2
                    break
            if test:pass
            else:
                train_set.append(sample)
                img[round(sample["a"]/2), sample["b"]] = 1
        os.mkdir(f"{title}")
        with open(f"{title}/train.json", "w") as f:
            json.dump(list(train_set), f)
        with open(f"{title}/test.json", "w") as f:
            json.dump(list(test_set), f)
        with open(f"{title}/test_squares.txt", "w") as f:
            f.write("centers:{}\nlengths:{}".format(centers, lengths))
        plt.imshow(img)
        plt.savefig(f"{title}/data_split")

'''
The script is to create the modular_addition(mod 113) dataset.

Here's an example:
1+113=1

we tokenize input as [tokenizer(1),tokenizer(+),tokenizer(113),tokenizer(=)

* test set: $a\in [50,60], \quad b\in[70,80]$
* train set: the rest
'''

import json
import numpy as np

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

if __name__ == "__main__":
    p = 113
    dataset = generate_dataset(np.arange(0,113), np.arange(0,113), p)
    test_a_range = (50, 60)
    test_b_range = (70, 80)
    train_samples = []
    test_samples = []
    for sample in dataset:
        if test_a_range[0] <= sample["a"] <= test_a_range[1] and test_b_range[0] <= sample["b"] <= test_b_range[1]:
            test_samples.append(sample)
        else:
            train_samples.append(sample)

    with open("train.json", "w") as f:
        json.dump(train_samples, f)

    with open("test.json", "w") as f:
        json.dump(test_samples, f)
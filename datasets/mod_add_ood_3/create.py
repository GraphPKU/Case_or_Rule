'''
The script is to create the modular_addition(mod 113) dataset.

Here's an example:
1+113=1

we tokenize input as [tokenizer(1),tokenizer(+),tokenizer(113),tokenizer(=)

* train set: $a\in [0,39]\cup[51,112], \quad b\in[0,79]\cup[91,112]$
* test set: $a\in [40,50], \quad b\in[80,90]$
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
    train_a_range = np.concatenate((np.arange(0,40), np.arange(51,113)),axis=0)
    train_samples = generate_dataset(train_a_range, train_b_range, p)
    test_a_range = np.arange(40,51)
    test_b_range = np.arange(80,91)
    test_samples = generate_dataset(test_a_range, test_b_range, p)

    with open("train.json", "w") as f:
        json.dump(train_samples, f)

    with open("test.json", "w") as f:
        json.dump(test_samples, f)
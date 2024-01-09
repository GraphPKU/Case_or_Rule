'''
The script is to create the modular_addition(mod 113) dataset.

Here's an example:
1+113=1

we tokenize input as [tokenizer(1),tokenizer(+),tokenizer(113),tokenizer(=)
'''

import json
import numpy as np

def generate_dataset(p, output_dir):
    samples = []
    idx = 0
    for a in range(p):
        for b in range(p):
            samples.append({
                "id": idx,
                "question": f"{a}+{b}=",
                "a": a,
                "b": b,
                "answers": [int(np.mod(a + b, p))]
                })
    print(len(samples))

    with open(output_dir, "w") as f:
        json.dump(samples, f)

if __name__ == "__main__":
    p = 113
    generate_dataset(p, f"mod_add.json")
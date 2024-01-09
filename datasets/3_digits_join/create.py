'''
The script is to create the first_3_digit dataset.

Here's an example:
1,0,1,1,0,1,1,1,0,1,0,0:0

We do XOR operation between the first three digits as the answer. In this case, 1 ^ 0 ^ 1 = 0, so the answer is 0.
'''

import json
import random
import itertools as it

def answer(q):
    return int(q[0]) ^ int(q[1]) ^ int(q[2])

def generate_dataset(num_q_digit, output_dir):
    questions = list(it.product(range(2), repeat=num_q_digit))
    samples = []
    for i, q in enumerate(questions):
        samples.append({
            "id": i,
            "question": ",".join([str(digit) for digit in q]) + ":",
            "answers": [str(answer(q))]
            })
    with open(output_dir, "w") as f:
        json.dump(samples, f)

if __name__ == "__main__":
    generate_dataset(12, "3_digits_join.json")
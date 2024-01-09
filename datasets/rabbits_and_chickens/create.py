'''
The script is to create the rabbits_and_chickens dataset.

Here's an example:
Q: Rabbits have 4 legs and 1 head. Chickens have 2 legs and 1 head. There are <<a>> legs and <<b>> heads on the farm. How many rabbits and chickens are there?
A: There are <<(a-2b)/2>> rabbits and <<(4b-a)/2>> chickens.
'''

import json
import numpy as np

def generate_dataset(min_num_head, max_num_head, output_dir):
    samples = []
    idx = 0
    question = "Rabbits have 4 legs and 1 head. Chickens have 2 legs and 1 head. There are {} legs and {} heads on the farm. How many rabbits and chickens are there?"
    answer = "There are {} rabbits and {} chickens."
    for heads in range(min_num_head, max_num_head):
        for legs in range(2 * heads, 4 * heads + 1, 2):
            samples.append({
                "id": idx,
                "question": question.format(legs, heads),
                "answers": [answer.format(int((legs - 2 * heads)/2), int((4 * heads - legs)/2))]
                })
            idx += 1
    print(len(samples))

    with open(output_dir, "w") as f:
        json.dump(samples, f)

if __name__ == "__main__":
    start = 0
    end = 100
    generate_dataset(start, end, f"rabbits_and_chickens.json")
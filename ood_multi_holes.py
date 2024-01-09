import torch
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
import transformers
from transformers import AutoTokenizer
from dataset import GPT2Dataset, GPT2DatasetReplace, TestDataset
from tqdm import tqdm
import numpy as np
import json
import re
import os
import matplotlib.pyplot as plt


device = "cuda:1"
model_name = "gpt2"
task = "add_ood"

cot = True
number = "a100_b100_5hole"

generation_num = 5

if cot:
    file_path = f"save_model_{model_name}/{task}/{number}_cot/model_99"
else:
    file_path = f"save_model_{model_name}/{task}/{number}/model_99"

print(f"loading model: {model_name}...")
model = GPT2LMHeadModel.from_pretrained(file_path).to(device)
tokenizer = GPT2Tokenizer.from_pretrained("pretrained_models/gpt2")
print("done!")

if cot:
    file_path_test=f'datasets/{task}/{number}/test_cot.json'
    file_path_train=f'datasets/{task}/{number}/train_cot.json'
else:
    file_path_test=f'datasets/{task}/{number}/test.json'
    file_path_train=f'datasets/{task}/{number}/train.json'

with open(file_path_test, "r") as f:
    test_set = json.load(f)

with open(file_path_train, "r") as f:
    train_set = json.load(f)

def extract_answer(s: str):
    return re.findall(r'[0-9]+\.?[0-9]*', s)[-1]

# def extract_answer_cot(rationale):
#     answer = rationale.split("\n")[-1].split(",")[1]
#     return answer

def generate(test_set, model, temperature=1, generation_num=10):
    '''
    return a list of dict [{
        "question":, "a":, "b":, "answers":[], "generated_answers":[], "acc":
    }]
    '''
    generated_samples = []
    for sample in tqdm(test_set):
        query = tokenizer(sample["question"], return_tensors="pt").input_ids
        if cot:
            answer = sample["gt"]
        else:
            answer = sample["answers"][0]
        generated_answers = []
        correct = 0
        all = 0
        for _ in range(generation_num):
            outputs = model.generate(query.to(device), max_length=50, do_sample=True, temperature=temperature, pad_token_id=50257)
            generated_answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            if all == 50:
                print("rationale: \n", generated_answer)
            # if cot:
            #     generated_answer = extract_answer_cot(generated_answer)
            # else:
            generated_answer = extract_answer(generated_answer)
            generated_answers.append(generated_answer)
            if all == 50:
                print("answer: ", generated_answer)
            all += 1
            if generated_answer == answer:
                correct += 1
        sample["generated_answers"] = generated_answers
        sample["accuracy"] = correct/all
        generated_samples.append(sample)
    return generated_samples

dataset = train_set + test_set
print(np.sqrt(len(dataset)))
results = generate(dataset, model, generation_num=generation_num)


a_range = b_range = 100
data = np.zeros((a_range, b_range))
for result in results:
    a = int(result["a"])
    b = int(result["b"])
    data[a][b] = result["accuracy"]


# save acc as np
np.save(f"{file_path}/acc.npy", data)

# save acc as fig
plt.title(f"{task}_{number}")
plt.imshow(data)
plt.colorbar()
plt.savefig(f"{file_path}/result.pdf")

# save generation num
with open(f"{file_path}/gen_num.txt", "w") as f:
    f.write(f"generation num: {generation_num}")

# if cot:
#     np.save(f"save_model_{model_name}/{task}/{number}_cot/model_99/acc.npy", data)
#     with open("")
# else:
#     np.save(f"save_model_{model_name}/{task}/{number}/model_99/acc.npy", data)
# plt.title(f"{task}_{number}")
# plt.imshow(data)
# plt.colorbar()
# if cot:
#     plt.savefig(f"save_model_{model_name}/{task}/{number}_cot/model_99/result.pdf")
# else:
#     plt.savefig(f"save_model_{model_name}/{task}/{number}/model_99/result.pdf")

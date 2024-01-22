import torch
from torch.autograd import grad
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
import transformers
from transformers import AutoTokenizer
from dataset import GPT2Dataset, GPT2DatasetReplace, TestDataset
from tqdm import tqdm
import random
import numpy as np
import json
import re
import os

device = "cuda:1"
task = "addition"
cot = False
title = "random_split_0.7_7000_3000_0-100"

print("loading model...")
model_name = "gpt2"
if cot:
    model = GPT2LMHeadModel.from_pretrained(f"save_model_{model_name}/{task}/{title}_cot/model_99").to(device)
else:
    model = GPT2LMHeadModel.from_pretrained(f"save_model_{model_name}/{task}/{title}/model_99").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("pretrained_models/gpt2")
print("done!")

if cot:
    file_path_test=f'datasets/{task}/{title}/test_cot.json'
    file_path_train=f'datasets/{task}/{title}/train_cot.json'
else:
    file_path_test=f'datasets/{task}/{title}/test.json'
    file_path_train=f'datasets/{task}/{title}/train.json'

with open(file_path_test, "r") as f:
    test_set = json.load(f)

with open(file_path_train, "r") as f:
    train_set = json.load(f)

def get_token_id(title: int):
    '''
    get the token_id for the title
    '''
    token_id = tokenizer(str(title)).input_ids
    assert len(token_id) == 1
    return token_id[0]

# we use the parameters corresponding to 0~99 in the last classification layer as \theta
w = model.lm_head.weight
token_ids = torch.tensor([get_token_id(n) for n in range(199)]).to(device)
theta = torch.index_select(w, 0, token_ids)

def hessian_vector_product(y,v):
    """
    Arguments:
        y: 标量/tensor，通常来说为loss函数的输出
        v: pytorch tensor的list，代表需要与hessian矩阵乘积的向量
    Returns:
        return_grads: pytorch tensor的list, 代表hvp的最终结果.
    Raises:
        ValueError: y 与 w 长度不同."""
    model.zero_grad()
    first_grads = grad(y, model.lm_head.weight, retain_graph=True, create_graph=True)[0]
    a = torch.index_select(first_grads, 0, token_ids).reshape(-1) @ v
    a.backward()
    return torch.index_select(model.lm_head.weight.grad, 0, token_ids).reshape(-1)

def extract_inputs_from_dataset(idx, dataset):
    sample = dataset[idx]["question"]+dataset[idx]["answers"][0]
    inputs = tokenizer(sample, return_tensors='pt').input_ids
    return inputs

def get_nabla_theta_L(z, alpha=1):
    r'''
    get $\nabla_{\theta}L(z, \theta)$
    '''
    model.zero_grad()
    outputs = model(z, labels=z)
    loss = outputs.loss
    loss = loss * alpha
    loss.backward()
    g = torch.index_select(model.lm_head.weight.grad, 0, token_ids)
    return g.reshape(-1)

def get_hv(test_idx, Hv=None, damp=0.01, alpha=1, iter_times=100, sample_num=1):
    model.zero_grad()
    z_test = extract_inputs_from_dataset(test_idx, test_set).to(device)
    v = get_nabla_theta_L(z_test, alpha).to(device)
    if Hv == None:
        Hv = v.to(device)
    Hv_list = []
    # start iteration
    # H_j^{-1}v = v + (I-\nabla^2_\theta L(z_{sj}, \theta))H_{j-1}^{-1}v
    for iter_time in tqdm(range(iter_times)):
        hvp = torch.zeros_like(v)
        # sample z_sj from training set
        for _ in range(sample_num):
            model.zero_grad()
            sj = random.randint(0, len(train_set)-1)
            z_sj = extract_inputs_from_dataset(sj, train_set).to(device)
            loss = model(z_sj, labels = z_sj).loss
            loss = loss * alpha
            hvp += hessian_vector_product(loss, Hv)
        Hv = v + (1-damp) * Hv - hvp / sample_num
        Hv_list.append(Hv[0].item())
        if iter_time % (iter_times/10) == 0:
            tqdm.write("="*20 + f" iter: {iter_time} "+"="*20)
            # print(Hv)
            tqdm.write(f"$\Delta$ = {torch.norm(v - hvp/sample_num)}")
    return Hv, Hv_list

def calc_IF(train_idx, Hv):
    z_train = extract_inputs_from_dataset(train_idx, train_set).to(device)
    nabla_theta_L_train = get_nabla_theta_L(z_train).to(device)
    return - Hv @ nabla_theta_L_train

test_idx = random.randint(0, len(test_set))
a_range = int(np.sqrt(len(test_set) + len(train_set)))
print(test_set[test_idx]["question"])

alpha = 0.02
iter_times = 100000
Hv, Hv_list = get_hv(test_idx, alpha=alpha, iter_times=iter_times)

I = np.zeros((a_range, a_range))
for train_idx in tqdm(range(len(train_set))):
    a = int(train_set[train_idx]["a"])
    b = int(train_set[train_idx]["b"])
    I[a, b] = calc_IF(train_idx, Hv)
# import matplotlib.pyplot as plt
# plt.imshow(I)
# plt.colorbar()
# plt.savefig(f"IF_{test_idx}.pdf")
np.save(f"save_model_{model_name}/{task}/{title}/model_99/{test_idx}_IF_array", I)
import matplotlib.pyplot as plt
plt.plot(np.arange(len(Hv_list)), Hv_list)
plt.savefig(f"save_model_{model_name}/{task}/{title}/model_99/{test_idx}_al={alpha}_it={iter_times}_test={test_idx}_d={0.01}.png")

import seaborn as sns; sns.set()

ax = sns.heatmap(I)
fig = ax.get_figure()
fig.savefig(f"save_model_{model_name}/{task}/{title}/model_99/{test_idx}_IF_{test_set[test_idx]['question']}.pdf")
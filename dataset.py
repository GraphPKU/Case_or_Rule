
import torch
from torch.utils.data import Dataset
from transformers.models.gpt2 import GPT2Tokenizer
import json
import numpy as np
import random

class GPT2Dataset(Dataset):
    def __init__(self, file_path: str, max_length: int, eda_aug: bool = False):
        self.file_path = file_path
        self.max_length = max_length
        self.tokenizer = GPT2Tokenizer.from_pretrained("pretrained_models/gpt2")
        # self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.eda_aug = eda_aug


        with open(self.file_path, "r") as f:
            self.json_file = json.load(f) # a list
        if 'answers' in self.json_file[0].keys():
            self.num_answer = [len(q['answers']) for q in self.json_file]
        elif 'answers_with_replace' in self.json_file[0].keys():
            self.num_answer = [len(q['answers_with_replace']) for q in self.json_file]
        else:
            raise ValueError("The json file must have the key 'answers' or 'answers_with_replace'.")
        self.cum_answer = np.cumsum(self.num_answer)

    def __len__(self):
        return self.cum_answer[-1]

    def __getitem__(self, idx):
        # first find the idx of answer, the first number in the self.cum_answer is larger than the idx.
        question_idx = np.searchsorted(self.cum_answer, idx, side='right')
        answer_idx = idx - self.cum_answer[question_idx-1] if question_idx > 0 else idx
        question = self.json_file[question_idx]['question']
        answer = self.json_file[question_idx]['answers'][answer_idx]
        text = f"{question}{answer}"
        # if self.eda_aug:
        #     from eda import eda
            # text = eda(text, alpha_sr=0.2, alpha_ri=0.1, alpha_rs=0, p_rd=0.1, num_aug=1)[0]
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        return encoding.input_ids.squeeze(), torch.tensor(0)

class GPT2DatasetReplace(GPT2Dataset):
    def __init__(self, file_path: str, max_length: int, replace_prob:float, random_seed: int, eda_aug: bool = False):
        super().__init__(file_path, max_length, eda_aug=eda_aug)
        self.replace_prob = replace_prob
        self.random_seed = random_seed
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        random.seed(self.random_seed)

    def __getitem__(self, idx):
        # first find the idx of answer, the first number in the self.cum_answer is larger than the idx.
        question_idx = np.searchsorted(self.cum_answer, idx, side='right')
        answer_idx = idx - self.cum_answer[question_idx-1] if question_idx > 0 else idx
        if np.random.rand() > self.replace_prob:
            question = self.json_file[question_idx]['question']
            answer = self.json_file[question_idx]['answers_with_replace'][answer_idx]['answer']
            text = question + answer
        else:
            # select one replacement in the self.json_file[question_idx]['answers_with_replace'][answer_idx]['replaced_qa'], which is a list
            text = random.choice(self.json_file[question_idx]['answers_with_replace'][answer_idx]['replaced_qa'])
        # if self.eda_aug:
        #     from eda import eda
            # text = eda(text, alpha_sr=0.2, alpha_ri=0.1, alpha_rs=0, p_rd=0.1, num_aug=1)[0]
        encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
        return encoding.input_ids.squeeze(), torch.tensor(0)

class TestDataset(Dataset):
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.tokenizer = GPT2Tokenizer.from_pretrained("pretrained_models/gpt2")
        # self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')

        with open(self.file_path, "r") as f:
            self.json_file = json.load(f) # a list

    def __len__(self):
        return len(self.json_file)
    def __getitem__(self, idx):
        return self.tokenizer(self.json_file[idx]['question'], return_tensors='pt').input_ids, self.json_file[idx]['answers'][0]
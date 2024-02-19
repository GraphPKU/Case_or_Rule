# This code is based on tatsu-lab/stanford_alpaca. Below is the original copyright:
#
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
from tqdm import tqdm
from dataclasses import dataclass, field
import json, os
import math
import pathlib
from typing import Dict, Optional, Sequence
import time, wandb
import numpy as np
import torch
from torch.utils.data import Dataset
import transformers
from transformers import Trainer
from transformers.trainer_pt_utils import LabelSmoother

from fastchat.conversation import SeparatorStyle
from fastchat.model.model_adapter import get_conversation_template

IGNORE_TOKEN_ID = LabelSmoother.ignore_index

timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
os.environ["WANDB__SERVICE_WAIT"] = "300"
wandb.init(project="Mod_ADD",name=timestamp)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    trust_remote_code: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to allow for custom models defined on the Hub in their own modeling files"
        },
    )
    padding_side: str = field(
        default="right", metadata={"help": "The padding side in tokenizer"}
    )


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def trainer_save_model_safe(trainer: transformers.Trainer):
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import StateDictType, FullStateDictConfig

    save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(
        trainer.model, StateDictType.FULL_STATE_DICT, save_policy
    ):
        trainer.save_model()


def preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conversation_template("llama-2")
    # conv = get_conversation_template("vicuna")

    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO
    assert conv.sep_style == SeparatorStyle.LLAMA2


    # Mask targets. Only compute loss on the assistant outputs.
    # seps = [conv.sep, conv.sep2]
    sep = conv.sep + conv.roles[1] + " "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)
            
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len + 2

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def ModAdd_preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conversation_template("llama-2")
    # conv = get_conversation_template("vicuna")

    # roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in tqdm(enumerate(sources)):
        
        conv.messages = []
        conv.append_message(conv.roles[0], source['question'])
        conv.append_message(conv.roles[1], source['answers'][0])
        # if roles[source[0]["from"]] != conv.roles[0]:
        #     # Skip the first one if it is not from human
        #     source = source[1:]

        # conv.messages = []
        # for j, sentence in enumerate(source):
        #     role = roles[sentence["from"]]
        #     assert role == conv.roles[j % 2], f"{i}"
        #     conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    # assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO
    assert conv.sep_style == SeparatorStyle.LLAMA2


    # Mask targets. Only compute loss on the assistant outputs.
    # seps = [conv.sep, conv.sep2]
    sep = conv.sep + conv.roles[1] + " "
    for conversation, target in tqdm(zip(conversations, targets)):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)
            
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len + 2

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )


def ModAdd_vicuna_preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # conv = get_conversation_template("llama-2")
    conv = get_conversation_template("vicuna")

    # roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        
        conv.messages = []
        conv.append_message(conv.roles[0], source['question'])
        conv.append_message(conv.roles[1], source['answers'][0])
        # if roles[source[0]["from"]] != conv.roles[0]:
        #     # Skip the first one if it is not from human
        #     source = source[1:]

        # conv.messages = []
        # for j, sentence in enumerate(source):
        #     role = roles[sentence["from"]]
        #     assert role == conv.roles[j % 2], f"{i}"
        #     conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()

    assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO
    # assert conv.sep_style == SeparatorStyle.LLAMA2


    # Mask targets. Only compute loss on the assistant outputs.
    # seps = [conv.sep, conv.sep2]
    # sep = conv.sep + conv.roles[1] + " "
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        for i, turn in enumerate(turns):
            if turn == "":
                break
            turn_len = len(tokenizer(turn).input_ids)
            
            parts = turn.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
            instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                instruction_len -= 1

            # Ignore the user instructions
            target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
            cur_len += turn_len 

            if i != 0 and not tokenizer.legacy:
                # The legacy and non-legacy modes handle special tokens differently
                cur_len -= 1

        target[cur_len:] = IGNORE_TOKEN_ID

        if False:  # Inspect and check the correctness of masking
            z = target.clone()
            z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
            rank0_print(tokenizer.decode(z))
            exit()

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_TOKEN_ID
                rank0_print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" #turn = {len(turns) - 1}. (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

def ModAdd_Unmask_preprocess(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    conv = get_conversation_template("llama-2")
    # conv = get_conversation_template("vicuna")

    # roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        
        conv.messages = []
        conv.append_message(conv.roles[0], source['question'])
        conv.append_message(conv.roles[1], source['answers'][0])
        # if roles[source[0]["from"]] != conv.roles[0]:
        #     # Skip the first one if it is not from human
        #     source = source[1:]

        # conv.messages = []
        # for j, sentence in enumerate(source):
        #     role = roles[sentence["from"]]
        #     assert role == conv.roles[j % 2], f"{i}"
        #     conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = tokenizer(
        conversations,
        return_tensors="pt",
        padding="max_length",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
    targets = input_ids.clone()


    # targets[:1] = IGNORE_TOKEN_ID
    # assert conv.sep_style == SeparatorStyle.ADD_COLON_TWO
    # assert conv.sep_style == SeparatorStyle.LLAMA2


    # # Mask targets. Only compute loss on the assistant outputs.
    # # seps = [conv.sep, conv.sep2]
    # sep = conv.sep + conv.roles[1] + " "
    for conversation, target in zip(conversations, targets):
        # total_len = int(target.ne(tokenizer.pad_token_id).sum())

        # turns = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_TOKEN_ID
        # for i, turn in enumerate(turns):
        #     if turn == "":
        #         break
        #     turn_len = len(tokenizer(turn).input_ids)
            
        #     parts = turn.split(sep)
        #     if len(parts) != 2:
        #         break
        #     parts[0] += sep
        #     # "-2" is hardcoded for the Llama tokenizer to make the offset correct.
        #     instruction_len = len(tokenizer(parts[0]).input_ids) - 2

        #     if i != 0 and not tokenizer.legacy:
        #         # The legacy and non-legacy modes handle special tokens differently
        #         instruction_len -= 1

        #     # Ignore the user instructions
        #     target[cur_len : cur_len + instruction_len] = IGNORE_TOKEN_ID
        #     cur_len += turn_len + 2

        #     if i != 0 and not tokenizer.legacy:
        #         # The legacy and non-legacy modes handle special tokens differently
        #         cur_len -= 1

        # target[cur_len:] = IGNORE_TOKEN_ID

        # if False:  # Inspect and check the correctness of masking
        #     z = target.clone()
        #     z = torch.where(z == IGNORE_TOKEN_ID, tokenizer.unk_token_id, z)
        #     rank0_print(tokenizer.decode(z))
        #     exit()

        # if cur_len < tokenizer.model_max_length:
        #     if cur_len != total_len:
        #         target[:] = IGNORE_TOKEN_ID
        #         rank0_print(
        #             f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
        #             f" #turn = {len(turns) - 1}. (ignored)"
        #         )

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )



class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, process_args):
        super(SupervisedDataset, self).__init__()

        rank0_print("Formatting inputs...")

        # sources = [example["conversations"] for example in raw_data]
        if process_args.pre_process_func == "vicuna":
            process_cls = ModAdd_vicuna_preprocess
        if process_args.pre_process_func == "Unmask":
            process_cls = ModAdd_Unmask_preprocess
        if process_args.pre_process_func == "None":
            process_cls = ModAdd_preprocess
        
        data_dict = process_cls(raw_data, tokenizer)
        # data_dict = ModAdd_preprocess(raw_data, tokenizer)

        self.input_ids = data_dict["input_ids"]
        self.labels = data_dict["labels"]
        self.attention_mask = data_dict["attention_mask"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            attention_mask=self.attention_mask[i],
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer, process_args):
        super(LazySupervisedDataset, self).__init__()
        self.tokenizer = tokenizer

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.raw_data = raw_data
        self.cached_data_dict = {}

        if process_args.pre_process_func == "vicuna":
            self.process_cls = ModAdd_vicuna_preprocess
        if process_args.pre_process_func == "Unmask":
            self.process_cls = ModAdd_Unmask_preprocess
        if process_args.pre_process_func == "None":
            self.process_cls = ModAdd_preprocess
        
        

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i in self.cached_data_dict:
            return self.cached_data_dict[i]

        # ret = preprocess([self.raw_data[i]["conversations"]], self.tokenizer)
        # ret = ModAdd_preprocess(self.raw_data, self.tokenizer)
        ret = self.process_cls(self.raw_data, self.tokenizer)
        
        ret = dict(
            input_ids=ret["input_ids"][0],
            labels=ret["labels"][0],
            attention_mask=ret["attention_mask"][0],
        )
        self.cached_data_dict[i] = ret

        return ret


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args, process_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = (
        LazySupervisedDataset if data_args.lazy_preprocess else SupervisedDataset
    )
    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    if len(train_json) > 500000:
        train_json = train_json[400000:500000]
    train_dataset = dataset_cls(train_json, tokenizer=tokenizer,process_args=process_args)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))[:500]
        eval_dataset = dataset_cls(eval_json, tokenizer=tokenizer, process_args=process_args)
    else:
        eval_dataset = None

    return dict(train_dataset=train_dataset, eval_dataset=eval_dataset)


# # revised by xiaojuan
# class ModAddDataset(Dataset):
#     """Dataset for supervised fine-tuning with mod addition."""


#     def __init__(self, raw_data, tokenizer: transformers.PreTrainedTokenizer):
#         super(SupervisedDataset, self).__init__()

#         rank0_print("Formatting inputs...")
#         sources = [example["conversations"] for example in raw_data]
#         data_dict = preprocess(sources, tokenizer)

#         self.input_ids = data_dict["input_ids"]
#         self.labels = data_dict["labels"]
#         self.attention_mask = data_dict["attention_mask"]

#     def __len__(self):
#         return len(self.input_ids)

#     def __getitem__(self, i) -> Dict[str, torch.Tensor]:
#         return dict(
#             input_ids=self.input_ids[i],
#             labels=self.labels[i],
#             attention_mask=self.attention_mask[i],
#         )

#     def __init__(self, file_path: str, max_length: int, eda_aug: bool = False):
#         self.file_path = file_path
#         self.max_length = max_length
#         self.tokenizer = GPT2Tokenizer.from_pretrained("pretrained_models/gpt2")
#         # self.tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-chat-hf')
#         self.tokenizer.pad_token = self.tokenizer.eos_token
#         # self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
#         self.eda_aug = eda_aug


#         with open(self.file_path, "r") as f:
#             self.json_file = json.load(f) # a list
#         if 'answers' in self.json_file[0].keys():
#             self.num_answer = [len(q['answers']) for q in self.json_file]
#         elif 'answers_with_replace' in self.json_file[0].keys():
#             self.num_answer = [len(q['answers_with_replace']) for q in self.json_file]
#         else:
#             raise ValueError("The json file must have the key 'answers' or 'answers_with_replace'.")
#         self.cum_answer = np.cumsum(self.num_answer)

#     def __len__(self):
#         return self.cum_answer[-1]

#     def __getitem__(self, idx):
#         # first find the idx of answer, the first number in the self.cum_answer is larger than the idx.
#         question_idx = np.searchsorted(self.cum_answer, idx, side='right')
#         answer_idx = idx - self.cum_answer[question_idx-1] if question_idx > 0 else idx
#         question = self.json_file[question_idx]['question']
#         answer = self.json_file[question_idx]['answers'][answer_idx]
#         text = f"{question}{answer}"
#         # if self.eda_aug:
#         #     from eda import eda
#             # text = eda(text, alpha_sr=0.2, alpha_ri=0.1, alpha_rs=0, p_rd=0.1, num_aug=1)[0]
#         encoding = self.tokenizer(text, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt")
#         return encoding.input_ids.squeeze(), torch.tensor(0)

# # revised by xiaojuan

def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    parser.add_argument(
        "--pre_process_func", type=str, required=True, help="how to process data"
    )
    model_args, data_args, training_args, process_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len and training_args.model_max_length > orig_ctx_len:
        scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
        config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side=model_args.padding_side,
        use_fast=False,
        trust_remote_code=model_args.trust_remote_code,
    )

    if tokenizer.pad_token != tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token

    # Load data
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, process_args=process_args)

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()

    # Save model
    model.config.use_cache = True
    trainer.save_state()
    if trainer.is_deepspeed_enabled:
        trainer.save_model()
    else:
        trainer_save_model_safe(trainer)


if __name__ == "__main__":
    
    train()

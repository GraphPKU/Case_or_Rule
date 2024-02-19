import os
import torch
from transformers.models.gpt2 import GPT2Config, GPT2LMHeadModel, GPT2Tokenizer
import transformers
from transformers import AutoTokenizer
from dataset import GPT2Dataset, GPT2DatasetReplace, TestDataset
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time
import re



task = "addition"
title = "1hole_(50, 50)_10_441_0-100"
model_name = "gpt2"
device = torch.device("cuda:0")
print(f"running {task} - {title}...")

save_model_path = f"save_model_{model_name}"
log_step = 200
num_epoch = 100
save_epoch = num_epoch / 5
batchsize = 30
lr = 1e-4
weight_decay= 0
random_seed = 42
torch.manual_seed(random_seed)
import random
random.seed(random_seed)

load_checkpoint = None
# tensorboard writer
timestamp = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
writer = SummaryWriter(log_dir='log/{}_{}_{}_{}'.format(model_name, task, title, timestamp))

# load pretrain model
print(f"loading pretrained model: {model_name}...")

model = GPT2LMHeadModel.from_pretrained(f"pretrained_models/{model_name}")
tokenizer = GPT2Tokenizer.from_pretrained(f"pretrained_models/{model_name}")
print("done")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",torch_dtype=torch.bfloat16, device_map="auto")
if task == "mod_addition" and "cot" in title:
    l = 100
else:
    l = 50
print(f"setting max length to {l}")
train_dataset = GPT2Dataset(file_path='datasets/{}/{}/train.json'.format(task,title), max_length=l)
valid_dataset = TestDataset(file_path='datasets/{}/{}/test.json'.format(task,title))
test_dataset = TestDataset(file_path='datasets/{}/{}/test.json'.format(task,title))

# train_dataset = GPT2Dataset(file_path='datasets/mod_add/train.json', max_length=50)
# valid_dataset = TestDataset(file_path='datasets/mod_add/test.json')
# test_dataset = TestDataset(file_path='datasets/mod_add/test.json')

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batchsize, shuffle=True)

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
# optimizer = torch.optim.SGD(model.parameters(), lr=lr)
num_training_steps = num_epoch * len(train_dataloader)
num_warmup_steps = 0.01 * num_training_steps
lr_scheduler = transformers.get_cosine_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)

progress_bar = tqdm(range(num_training_steps))

# def extract_answer(s: str):
#     return re.findall(r'[0-9]+\.?[0-9]*', s)

def extract_answer(s: str, mode=task):
    if mode in ["addition", "mod_addition", "base_addition", "linear_regression"]:
        return re.findall(r'[0-9]+\.?[0-9]*', s)[-1]

    elif mode in ["chickens_and_rabbits"]:
        return re.findall(r'[0-9]+\.?[0-9]*', s)[-2:]

    elif mode in ["addition_code"]:
        try:
            l = eval(re.findall(r'\[.+\]', s)[-1])
            return l
        except:
            print(s)
            return None

def valid_and_test(model, valid_dataset, test_dataset, device, step):
    with torch.no_grad():
        model.eval()
        valid_correct = 0
        ctr = 0
        for valid_question, valid_answer in tqdm(valid_dataset):
            valid_answer = extract_answer(str(valid_answer))
            outputs = model.generate(valid_question.to(device), max_length=l, num_beams=1, do_sample=False, pad_token_id=50257) # no padding, greedy decoding
            generated_answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
            # tqdm.write('-'*40)
            # tqdm.write(generated_answer)
            # tqdm.write('### The groundtruth is {}'.format(valid_answer))
            if ctr % 500 == 0:
                tqdm.write('-'*40)
                tqdm.write(generated_answer)
                tqdm.write('### The groundtruth is {}'.format(valid_answer))
            generated_answer = generated_answer[len(tokenizer.decode(valid_question.squeeze())):]
            generated_answer = extract_answer(generated_answer)
            if generated_answer is None:
                continue
            if generated_answer == valid_answer:
                valid_correct += 1
            ctr += 1
        tqdm.write('valid accuracy: {}, num of valid samples: {}'.format(valid_correct/(len(valid_answer) * len(valid_dataset)), len(valid_dataset)))
        writer.add_scalar('valid_accuracy', valid_correct/len(valid_dataset), step)

# main loop
step = 0
for epoch in range(num_epoch):
    optimizer.zero_grad()
    model.to(device)
    model.train()
    for batch in train_dataloader:
        batch = batch[0] # the labels is not used, because the labels is the same as the input_ids
        labels = batch
        # labels = torch.concatenate((batch[:, 1:], 50256*torch.ones(batch.shape[0], 1)), dim=1).long().to(device)
        # labels = batch[1:]
        # add the eot before the first padding token
        # for i in range(batch.shape[0]):
        #     for j in range(batch.shape[1]):
        #         if batch[i, j] == tokenizer.pad_token_id:
        #             batch[i, j] = tokenizer.eos_token_id
        #             break
        outputs = model(batch.to(device), labels=labels.to(device))
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        writer.add_scalar('loss', loss.item(), progress_bar.n)
        writer.add_scalar('lr', lr_scheduler.get_last_lr()[0], progress_bar.n)
        if progress_bar.n % 600 == 0:
            valid_and_test(model, valid_dataset, test_dataset, device, step=progress_bar.n)
        if step % log_step == 0:
            tqdm.write('epoch {}, step {}, loss {}, lr: {}'.format(epoch, progress_bar.n, loss.item(), lr_scheduler.get_last_lr()[0]))
        step += 1

GPT2LMHeadModel.save_pretrained(model, os.path.join(save_model_path, task, title, f"model_{epoch}"))

"""Generate answers with local models.

Usage:
python3 gen_model_answer.py --model-path lmsys/fastchat-t5-3b-v1.0 --model-id fastchat-t5-3b-v1.0
"""
import argparse
import json
import os
import random
import time

import shortuuid
import torch
from tqdm import tqdm

from fastchat.model import load_model, get_conversation_template
from fastchat.utils import str_to_torch_dtype
from typing import Optional
import re,ast


from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

def load_json(question_file:str, begin: Optional[int], end: Optional[int]):
    with open(question_file, "r") as f:
        questions = json.load(f)
        return questions[begin:end]
    
def extract_answer(s: str):
    return re.findall(r'[0-9]+\.?[0-9]*', s)[-1]
# def extract_answer_code(s: str):
#     return s.lower().split("final result:")[-1].strip()

# def extract_answer_code(s: str):
#     PATTERN = r'result=(\[[^\]]*])'
#     return re.findall(PATTERN, s)[-1]

def extract_answer_code(s: str):
    # PATTERN = "(?<=4\. Return Result\\n```\nreturn result\\n```\nresult=)\[.*?\]"
    PATTERN = r'result=(\[[^\]]*])'
    # s = s.lower().split("backtracking")[0]
    s = s.lower().split("end")[0]
    return re.findall(PATTERN, s)[-1]

def run_eval(
    model_path,
    model_id,
    question_file,
    question_begin,
    question_end,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    num_gpus_total,
    max_gpu_memory,
    dtype,
    revision,
    answers,
):
    questions = load_json(question_file, question_begin, question_end)
    # random shuffle the questions to balance the loading
    random.shuffle(questions)

    # Split the question file into `num_gpus` files
    assert num_gpus_total % num_gpus_per_model == 0
    use_ray = num_gpus_total // num_gpus_per_model > 1

    if use_ray:
        get_answers_func = ray.remote(num_gpus=num_gpus_per_model)(
            get_model_answers
        ).remote
    else:
        get_answers_func = get_model_answers

    chunk_size = len(questions) // (num_gpus_total // num_gpus_per_model)
    ans_handles = []
    for i in range(0, len(questions), chunk_size):
        ans_handles.append(
            get_answers_func(
                model_path,
                model_id,
                questions[i : i + chunk_size],
                answer_file,
                max_new_token,
                num_choices,
                num_gpus_per_model,
                max_gpu_memory,
                dtype=dtype,
                revision=revision,
                answers=answers
            )
        )

    if use_ray:
        ray.get(ans_handles)

@torch.inference_mode()
def get_model_answers(
    model_path,
    model_id,
    questions,
    answer_file,
    max_new_token,
    num_choices,
    num_gpus_per_model,
    max_gpu_memory,
    dtype,
    revision,
    answers,
    generated_num=1,
):
    model, tokenizer = load_model(
        model_path,
        revision=revision,
        device="cuda",
        num_gpus=num_gpus_per_model,
        max_gpu_memory=max_gpu_memory,
        dtype=dtype,
        load_8bit=False,
        cpu_offloading=False,
        debug=False,
    )
    model = model.merge_and_unload()

    gen_all_acc = 0.0
    num_id = 0
    for question in tqdm(questions):
        # if question["category"] in temperature_config:
        #     temperature = temperature_config[question["category"]]
        # else:
        #     temperature = 0.7
        if question['id'] in answers.keys():
            continue
        temperature = 0.0

        conv = get_conversation_template(model_id)
        # conv.set_system_message("")
        # q_text = f"({question['question'][:-1]})%113="
        q_text = question['question']
        # conv.set_system_message("You are a helpful, respectful and honest assistant with mathematical abilities. Please compute the sum of the two numbers. Please answer with final result.")
        conv.append_message(conv.roles[0],q_text) 
        # conv.append_message(conv.roles[0],"Hello!") 

        # conv.append_message(conv.roles[0],sample['conversations'][0]['value'] + " Let's think step by step. ") 

        conv.append_message(conv.roles[1], None)
        
        prompt = conv.get_prompt()
        # 
        # prompt = q_text
        print(prompt)
        input_ids = tokenizer([prompt]).input_ids

        if temperature < 1e-4:
            do_sample = False
        else:
            do_sample = True

        generated_answers = []
        gen_raw_output = []
        all = 0
        correct = 0
        for _ in range(generated_num):

            # some models may error out when generating long outputs
            try:
                output_ids = model.generate(
                    torch.as_tensor(input_ids).cuda(),
                    do_sample=do_sample,
                    temperature=temperature,
                    max_new_tokens=max_new_token,
                )
                if model.config.is_encoder_decoder:
                    output_ids = output_ids[0]
                else:
                    output_ids = output_ids[0][len(input_ids[0]) :]

                # be consistent with the template's stop_token_ids
                if conv.stop_token_ids:
                    stop_token_ids_index = [
                        i
                        for i, id in enumerate(output_ids)
                        if id in conv.stop_token_ids
                    ]
                    if len(stop_token_ids_index) > 0:
                        output_ids = output_ids[: stop_token_ids_index[0]]

                output = tokenizer.decode(
                    output_ids,
                    spaces_between_special_tokens=False,
                )
                if conv.stop_str and isinstance(conv.stop_str, list):
                    stop_str_indices = sorted(
                        [
                            output.find(stop_str)
                            for stop_str in conv.stop_str
                            if output.find(stop_str) > 0
                        ]
                    )
                    if len(stop_str_indices) > 0:
                        output = output[: stop_str_indices[0]]
                elif conv.stop_str and output.find(conv.stop_str) > 0:
                    output = output[: output.find(conv.stop_str)]

                for special_token in tokenizer.special_tokens_map.values():
                    if isinstance(special_token, list):
                        for special_tok in special_token:
                            output = output.replace(special_tok, "")
                    else:
                        output = output.replace(special_token, "")
                output = output.strip()
                gen_raw_output.append(output)
                if conv.name == "xgen" and output.startswith("Assistant:"):
                    output = output.replace("Assistant:", "", 1).strip()

                try:
                    
                    # generated_answer = extract_answer(output)
                    # generated_answers.append(generated_answer)
                    # gt = extract_answer(question['answers'][0])
                    # if float(generated_answer) == float(gt):
                    #     correct += 1
                    
                    generated_answer = extract_answer_code(output)
                    generated_answers.append(generated_answer)
                    gt = extract_answer_code(question['answers'][0])
                    if ast.literal_eval(generated_answer) == ast.literal_eval(gt) :
                        correct += 1

                except Exception as e:
                    # print(e)
                    # print(output)
                    generated_answers.append("error")

                
                all += 1
                

            except RuntimeError as e:
                print("ERROR question ID: ", question["id"])
                output = "ERROR"

            

        # Dump answers
        os.makedirs(os.path.dirname(answer_file), exist_ok=True)
        with open(os.path.expanduser(answer_file), "a") as fout:
            ans_json = {
                "question_id": question["id"],
                "accuracy": correct / all,
                "question": q_text,
                "a": question["a"],
                "b": question["b"],
                "generated_answers": generated_answers,
                "gen_raw_output": gen_raw_output,
                "tstamp": time.time(),
            }
            fout.write(json.dumps(ans_json) + "\n")
        gen_all_acc += correct / all
        num_id += 1
        print("acc:", gen_all_acc / num_id)
        

def reorg_answer_file(answer_file):
    """Sort by question id and de-duplication"""
    answers = {}
    with open(answer_file, "r") as fin:
        for l in fin:
            qid = json.loads(l)["question_id"]
            answers[qid] = l

    qids = sorted(list(answers.keys()))
    with open(answer_file, "w") as fout:
        for qid in qids:
            fout.write(answers[qid])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # device = "cuda:0"
    # task = "mod_add_ood_multi_holes"
    # number = "5"
    parser.add_argument(
        "--task",
        type=str,
        default="mod_add_ood_multi_holes",
        help="The task name"
    )
    parser.add_argument(
        "--number",
        type=str,
        default="5",
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        # required=True,
        help="The path to the weights. This can be a local folder or a Hugging Face repo ID.",
    )
    parser.add_argument(
        "--model-id", type=str, default="llama-2", help="A custom name for the model."
    )
    # parser.add_argument(
    #     "--bench-name",
    #     type=str,
    #     default="mt_bench",
    #     help="The name of the benchmark question set.",
    # )
    parser.add_argument(
        "--question-begin",
        type=int,
        help="A debug option. The begin index of questions.",
    )
    parser.add_argument(
        "--question-end", type=int, help="A debug option. The end index of questions."
    )
    parser.add_argument("--answer-file", type=str, help="The output answer file.")
    parser.add_argument(
        "--max-new-token",
        type=int,
        default=4096,
        help="The maximum number of new generated tokens.",
    )
    parser.add_argument(
        "--num-choices",
        type=int,
        default=1,
        help="How many completion choices to generate.",
    )
    parser.add_argument(
        "--num-gpus-per-model",
        type=int,
        default=1,
        help="The number of GPUs per model.",
    )
    parser.add_argument(
        "--num-gpus-total", type=int, default=4, help="The total number of GPUs."
    )
    parser.add_argument(
        "--validation",
        type=str,
        default = "test"
    )
    parser.add_argument(
        "--max-gpu-memory",
        type=str,
        help="Maxmum GPU memory used for model weights per GPU.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float16", "bfloat16"],
        help="Override the default dtype. If not set, it will use float16 on GPU and float32 on CPU.",
        default=None,
    )
    parser.add_argument(
        "--revision",
        type=str,
        default="main",
        help="The model revision to load.",
    )
    parser.add_argument(
        "--question-file",
        type=str,
        default=""
    )
    parser.add_argument(
        "--test-id",
        type=str,
        default='test_8'
    )

    args = parser.parse_args()

    if args.num_gpus_total // args.num_gpus_per_model > 1:
        import ray

        ray.init()

    question_file = f"{args.question_file}/{args.test_id}.json"
    if args.answer_file:
        answer_file = args.answer_file
    else:
        if "checkpoint" in args.model_path.split('/')[-1]:
            model_path = args.model_path.split('/')[-2] + "/" + args.model_path.split('/')[-1]
        else:
            model_path = args.model_path.split('/')[-1]
        answer_file = f"{args.question_file}/model_answer/{model_path}_{args.test_id}.jsonl"
    
    answers = dict()
    if os.path.exists(answer_file):
        with open(answer_file, "r") as fin:
            for l in fin:
                qid = json.loads(l)["question_id"]
                answers[qid] = l
        # os.remove(answer_file)

    print(f"Output to {answer_file}")

    run_eval(
        model_path=args.model_path,
        model_id=args.model_id,
        question_file=question_file,
        question_begin=args.question_begin,
        question_end=args.question_end,
        answer_file=answer_file,
        max_new_token=args.max_new_token,
        num_choices=args.num_choices,
        num_gpus_per_model=args.num_gpus_per_model,
        num_gpus_total=args.num_gpus_total,
        max_gpu_memory=args.max_gpu_memory,
        dtype=str_to_torch_dtype(args.dtype),
        revision=args.revision,
        answers = answers
    )

    reorg_answer_file(answer_file)

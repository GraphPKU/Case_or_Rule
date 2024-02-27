import openai
import re
import random
from tqdm import tqdm
import math
from typing import Literal

api_key = ''
random_seed = 2024021
random.seed(random_seed)
import numpy as np
np.random.seed(random_seed)
BASE = 9

class BaseInt(int):
    def __new__(cls, value: int) -> 'BaseInt':
        for s in str(value):
            if int(s) >= BASE:
                raise ValueError(f"The value {value} is not a valid base {BASE} integer.")
        return super().__new__(cls, value)
    
    @staticmethod
    def _from_base_10_int(value: int) -> "BaseInt":
        # value is a base 10 int, convert it into Base 
        if value < 0:
            raise ValueError("The value should be a non-negative integer.")
        if value == 0:
            return BaseInt(0)
        res = ""
        while value > 0:
            res = str(value % BASE) + res
            value //= BASE
        return BaseInt(int(res))
    
    @staticmethod
    def random_generate_length(length: int) -> "BaseInt":
        if length <= 0:
            raise ValueError("The length should be a positive integer.")
        return BaseInt(int(''.join([str(random.randint(1, BASE-1))] + [str(random.randint(0, BASE-1)) for _ in range(length-1)])))
    
    def _to_base_10_int(self) -> int:
        return int(str(self), BASE)
    
    def __add__(self, another: "BaseInt") -> "BaseInt":
        if not isinstance(another, BaseInt):
            raise TypeError("The addition should only be performed between two BaseInt objects.")
        return BaseInt._from_base_10_int(self._to_base_10_int() + another._to_base_10_int())
    
    def __sub__(self, another: "BaseInt") -> "BaseInt":
        return BaseInt._from_base_10_int(self._to_base_10_int() - another._to_base_10_int())
    
def is_easy_add(a: BaseInt, b: BaseInt) -> bool:
    """
    Return whether there is no carry in the (base) addition of a and b.
    """
    return all([int(digit) + int(b_digit) < BASE for digit, b_digit in zip(str(a)[::-1], str(b)[::-1])])
    
def gen_cot_rationale(a: BaseInt, b: BaseInt):
    '''
    return the scratchpad rationale for the question
    '''
    
    def count_c(a_digit: str, b_digit: str, c:BaseInt) -> BaseInt:
        if int(a_digit) + int(b_digit) + c._to_base_10_int() >= BASE:
            return BaseInt(1)
        else:
            return BaseInt(0)
    
    a_digits: list[str] = [digit for digit in str(a)]
    b_digits: list[str] = [digit for digit in str(b)]
    rationale = ""
    answer = ""
    c: BaseInt = BaseInt(0)
    for i in range(len(str(a+b))+1):
        line = f"{''.join(a_digits)}+{''.join(b_digits)},{answer},C:{c}\n"
        a_digit = a_digits[-1] if a_digits else str(0)
        b_digit = b_digits[-1] if b_digits else str(0)
        if i != len(str(a+b)):
            adding = f"# added {a_digit}+{b_digit}+{c}={str(BaseInt(int(a_digit)) + BaseInt(int(b_digit)) + c)[-1]}\n"
            line += adding
        rationale += line
        if a_digits or b_digits:
            answer = str(BaseInt(int(a_digit)) + BaseInt(int(b_digit)) + c)[-1] + answer
            c = count_c(a_digit, b_digit, c)
            if a_digits:
                a_digits.pop()
            if b_digits:
                b_digits.pop()
        else:
            if c:
                answer = str(c) + answer
            c = BaseInt(0)
    rationale = rationale.strip()
    return f"{rationale}"

def gen_rf_rationale(a: BaseInt, b: BaseInt):
    import prompt_base9 as prompt
    def rationale(a_s: list[int], b_s: list[int]):
        rationale = prompt.NUM.format(a_s, b_s) + prompt.INITIALIZE
        # Initialize the result list and carry
        result = []
        carry = 0
        # Loop through each digit
        while a_s or b_s:
            rationale += prompt.CHECK_THE_STOP_CRITERION_2_1_ENTER.format(a_s, b_s, bool(a_s), bool(b_s), bool(a_s) or bool(b_s)) + prompt.ONE_ITERATION_2_2
            # pop a
            rationale += prompt.POP_DIGIT.format(1, a_s, bool(a_s), a_s[:-1], a_s[-1]) if a_s else prompt.NO_POP_DIGIT.format(1)
            digit1 = a_s.pop() if a_s else 0
            # pop b
            rationale += prompt.POP_DIGIT.format(2, b_s, bool(b_s), b_s[:-1], b_s[-1]) if b_s else prompt.NO_POP_DIGIT.format(2)
            digit2 = b_s.pop() if b_s else 0
            # calculate total, result, carry
            # Calculate the sum of the current digits and the carry
            total = digit1 + digit2 + carry
            rationale += prompt.TOTAL_RESULT_CARRY.format(digit1, digit2, carry, total, result, total, total % BASE, [total%BASE] + result, total, total // BASE)
            # Insert the last digit of total to the result and update carry
            result.insert(0, total % BASE)
            carry = total // BASE
        rationale += prompt.CHECK_THE_STOP_CRITERION_2_1_END.format(a_s, b_s, bool(a_s), bool(b_s), bool(a_s) or bool(b_s))
        # If there's a remaining carry, insert it to the result
        rationale += prompt.CHECK_REMAINING_CARRY_TRUE.format(result, [1] + result) if carry else prompt.CHECK_REMAINING_CARRY_FALSE.format(result)
        if carry:
            result.insert(0, carry)
        # Return the result
        rationale += prompt.RETURN_THE_RESULT.format(result) + "\n" + str(result)
        return rationale
    
    return rationale([int(d) for d in str(a)], [int(d) for d in str(b)])

def get_response_from_gpt_api(text: str) -> str:
    # Get the response from the GPT-3.5 API
    client = openai.OpenAI(api_key=api_key)
    try_num = 5
    for _ in range(try_num):
        try:
            response = client.chat.completions.create(
                model='gpt-3.5-turbo',
                messages=[
                    {"role": "system", "content": f"You are a helpful assistant to solve arithmetic problems. You will be provided with two base {BASE} integers and you need to return the sum of the two integers in base {BASE}.\n"},
                    {"role": "user", "content":text},
                ],
            )
            if response.choices[0].message.content is None:
                raise ValueError("No content in the response")
            break
        except Exception as e:
            print(e)
            continue
    else:
        return ""
    return response.choices[0].message.content

def extract_answer_from_text(text: str) ->  None | int:
    # Find the last integer (considering possible punctuations) from the response
    matches = re.findall(r"(\d+(,\d+)*)", text)
    if matches:
        return int(matches[-1][0].replace(",", ""))
    else:
        return None
    
def generate_text_a_b(a: BaseInt, b: BaseInt, type: Literal["direct", "scratchpad", "rf"] = "direct", include_answer: bool = False) -> str:
    if type == "direct":
        if include_answer:
            return f"Int a: {a}; Int b: {b}. The sum of a and b is: a+b={a+b}.\n"
        else:
            return f"Directly generate the answer without solving steps. Int a: {a}; Int b: {b}. The sum of a and b is: a+b="
    elif type == "scratchpad":
        if include_answer:
            return f"Int a: {a}; Int b: {b}.\n{gen_cot_rationale(a, b)}\nTherefore, the answer is {a+b}.\n"
        else:
            return f"Int a: {a}; Int b: {b}.\n"
    elif type == "rf":
        if include_answer:
            return "-"*10+"\n"+f"Int a: {a}; Int b: {b}.\n{gen_rf_rationale(a, b)}\nTherefore, the answer is {a+b}.\n"
        else:
            return f"Int a: {a}; Int b: {b}.\n"
    else:
        raise ValueError("Invalid type")
    
def test_a_b(a:BaseInt, b:BaseInt, few_shot: None | list[tuple[BaseInt, BaseInt]] = None, test_num: int = 1, detail: bool = False, type: Literal["direct", "scratchpad", "rf"]='direct') -> float:
    from copy import deepcopy
    few_shot = deepcopy(few_shot) if few_shot is not None else None
    
    icl_prompt = ''
    if few_shot is None:
        pass
    else:
      # icl_prompt += "There are some examples:\n"
      random.shuffle(few_shot)
      for icl_a, icl_b in few_shot:
        icl_prompt += generate_text_a_b(icl_a, icl_b, type=type, include_answer=True)  
    
    input_text = icl_prompt + generate_text_a_b(a, b, type=type, include_answer=False)
    if detail:
        print('\n'.join(['-'*30, 'Input text:', input_text, '-'*30]))
    answers = []
    for _ in range(test_num):
        generated_text = get_response_from_gpt_api(input_text)
        if detail:
            print(generated_text)
        answers.append(extract_answer_from_text(generated_text))
    label = a + b
    
    correct = 0
    for answer in answers:
        if label == answer:
            correct += 1
            
    return correct / test_num
    
def generate_test_sample(max_length: int, save_name: str, test_sample: None | list[tuple[BaseInt, BaseInt]] = None, max_api_num: int = 100, target_num: int = 10, another_length: Literal['shorter', 'same', 'both'] = 'same') -> None:
    """use GPT api to filter some test pairs with accu less than 0.4."""
    if test_sample is None: 
        test_sample = []
    for _ in tqdm(range(max_api_num)):
        if another_length == 'shorter':
            length = random.randint(1, max_length-1)
        elif another_length == 'both':
            length = random.randint(1, max_length)
        elif another_length == 'same':
            length = max_length
        else:
            raise ValueError('Invalid another_length.')
        a = BaseInt.random_generate_length(max_length)
        b = BaseInt.random_generate_length(length)
        if random.random() < 0.5:
            a, b = b, a
        accu = test_a_b(a, b, detail=False, test_num=10)
        if accu < 0.2:
            test_sample.append((a, b))
        if len(test_sample) >= target_num:
            break
    print(f'Generate {len(test_sample)} test samples.')
    with open(save_name, 'wb') as wf:
        import pickle
        pickle.dump(test_sample, wf)
    
def generate_random_icl_sample(num: int, a_length: int, b_length: int, remove_similar: bool=False, orig_a:BaseInt|None=None, orig_b:BaseInt|None=None, similar_type: Literal['replace', 'keep', 'near']|None=None, similar_num_digit: int|None=None, similar_range: int|None = None) -> list[tuple[BaseInt, BaseInt]]:
    def issimilar(a,b):
        if similar_type == 'replace':
            assert similar_num_digit is not None
            nonreplace_a = []
            for i in range(len(str(a))):
                if str(a)[i] == str(orig_a)[i]:
                    nonreplace_a.append(len(str(a))-i-1)
            nonreplace_b = []
            for i in range(len(str(b))):
                if str(b)[i] == str(orig_b)[i]:
                    nonreplace_b.append(len(str(b))-i-1)
            if len(set(nonreplace_a).intersection(set(nonreplace_b))) >= min(len(str(orig_a)), len(str(orig_b))) - similar_num_digit:
                return True
            else:
                return False
            
        if similar_type == 'keep':
            assert similar_num_digit is not None
            nonreplace_a = []
            for i in range(len(str(a))):
                if str(a)[i] == str(orig_a)[i]:
                    nonreplace_a.append(len(str(a))-i-1)
            nonreplace_b = []
            for i in range(len(str(b))):
                if str(b)[i] == str(orig_b)[i]:
                    nonreplace_b.append(len(str(b))-i-1)
            if len(set(nonreplace_a).intersection(set(nonreplace_b))) >= similar_num_digit:
                return True
            else:
                return False
            
        if similar_type == 'near':
            assert similar_range is not None
            assert orig_a is not None and orig_b is not None
            if a in [orig_a - BaseInt(i) for i in range(1, similar_range+1)] + [orig_a]+[orig_a + BaseInt(i) for i in range(1, similar_range+1)] and b in [orig_b - BaseInt(i) for i in range(1, similar_range+1)] +[orig_b]+ [orig_b + BaseInt(i) for i in range(1, similar_range+1)]:
                return True
            else:
                return False
        
    if remove_similar: assert orig_a is not None and orig_b is not None and similar_type is not None and (similar_range is not None or similar_num_digit is not None)
    test_sample = []
    for _ in range(num):
        while True:
            a = BaseInt.random_generate_length(a_length)
            b = BaseInt.random_generate_length(b_length)
            if orig_a is not None and orig_b is not None:
                if a+b == orig_a+orig_b:
                    continue
                if a == orig_a:
                    continue
                if b == orig_b:
                    continue
            if remove_similar and issimilar(a, b):
                continue
            elif (a, b) in test_sample:
                continue
            else:
                break
        
        test_sample.append((a, b))
    return test_sample

def generate_similar_icl_sample(num: int, a: BaseInt, b: BaseInt, similar: Literal['replace', 'keep', 'near'], num_digit: int = 3, srange: int = 10, sure_every_digit: bool = False) -> list[tuple[BaseInt, BaseInt]]:
    if sure_every_digit:
        assert similar == 'replace' and num_digit == 1
    str_a = [s for s in str(a)]
    str_b = [s for s in str(b)]
    test_sample = []
    
    min_length = min(len(str_a), len(str_b))
    
    if similar == 'near':
        new_a_candidate: list[BaseInt] = []
        new_b_candidate: list[BaseInt] = []
        for d in range(1, srange):
            new_a_candidate.append(a - BaseInt(d))
            new_b_candidate.append(b - BaseInt(d))
            new_a_candidate.append(a + BaseInt(d))
            new_b_candidate.append(b + BaseInt(d))
    if sure_every_digit:
        record_digit = []
    for _ in range(num):
        while True:
            left_short = random.randint(0, min_length-num_digit) if min_length > num_digit else 0
            if sure_every_digit and (left_short not in record_digit): # type: ignore
                record_digit.append(left_short)
            left_a = len(str_a) - min_length + left_short
            left_b = len(str_b) - min_length + left_short
            right_a = min(left_a + num_digit, len(str_a))
            right_b = min(left_b + num_digit, len(str_b))
            if similar == 'replace':
                new_a = BaseInt(int(''.join(str_a[:left_a] + [str(random.randint(0, BASE-1)) for _ in range(right_a - left_a)] + str_a[right_a:])))
                new_b = BaseInt(int(''.join(str_b[:left_b] + [str(random.randint(0, BASE-1)) for _ in range(right_b - left_b)] + str_b[right_b:])))
            elif similar == 'keep':
                new_a = BaseInt(int(''.join([str(random.randint(0,BASE-1)) for _ in range(left_a)] + str_a[left_a:right_a] + [str(random.randint(0,BASE-1)) for _ in range(len(str_a)-right_a)])))
                new_b = BaseInt(int(''.join([str(random.randint(0,BASE-1)) for _ in range(left_b)] + str_b[left_b:right_b] + [str(random.randint(0,BASE-1)) for _ in range(len(str_b)-right_b)])))
            elif similar == 'near':
                new_a = random.choice(new_a_candidate)
                new_b = random.choice(new_b_candidate)
            else:
                raise ValueError("Invalid similar type")
            if new_a + new_b != a + b and new_a != a and new_b != b and (new_a, new_b) not in test_sample:
                break
        test_sample.append((new_a, new_b))
    if sure_every_digit and len(record_digit) <= min_length-num_digit:
        return generate_similar_icl_sample(num=num, a=a, b=b, similar=similar, num_digit=num_digit, srange=srange, sure_every_digit=sure_every_digit)
    return test_sample

def sample_remove_index(num_example: int, num_mask: None | int = None, num_remove: None | int= None, similar_index: list[int] | None = None) -> list[list[Literal[0, 1]]]:
    if similar_index is None:
        return _sample_remove_index(num_example=num_example, num_mask=num_mask, num_remove=num_remove)
    else:
        if num_mask is None:
            similar_num = 2 ** len(similar_index) - 1
            random_num = 2 ** (num_example - len(similar_index)) - 1
        else:
            similar_num = sum([random.random() < 0.5 for _ in range(num_mask)])
            random_num = num_mask - similar_num
        inner_index = _sample_remove_index(num_example=len(similar_index), num_mask=similar_num, num_remove=num_remove)
        return_list = []
        for part_mask in inner_index:
            mask = [1 if (i not in similar_index or part_mask[similar_index.index(i)] == 1) else 0 for i in range(num_example)]
            return_list.append(mask)
        
        similar_index = list(set(range(num_example)) - set(similar_index))
        inner_index = _sample_remove_index(num_example=len(similar_index), num_mask=random_num, num_remove=num_remove)
        for part_mask in inner_index:
            mask = [1 if (i not in similar_index or part_mask[similar_index.index(i)] == 1) else 0 for i in range(num_example)]
            return_list.append(mask)
        return return_list

def _sample_remove_index(num_example: int, num_mask: None | int = None, num_remove: None | int= None) -> list[list[Literal[0, 1]]]:
    return_list = []
    
    if num_remove is None:
        if num_mask is None or num_mask > 2 ** num_example - 1:
            num_mask = 2 ** num_example - 1
            masks_s = range(0, 2**num_example-1)
        else:
            masks_s = random.sample(population=range(0, 2**num_example-1), k=num_mask)
        for mask_s in masks_s:
            s = f'{mask_s:b}'
            s = f'{s:0>{num_example}}'
            return_list.append(list(map(lambda x: int(x), s)))
    else:
        num_remove = min(num_remove, num_example)
        if num_mask is None or num_mask > math.comb(num_example, num_remove):
            num_mask = math.comb(num_example, num_remove)
        masks_s = range(0, 2**num_example-1)
        for mask_s in masks_s:
            s = f'{mask_s:b}'
            s = f'{s:0>{num_example}}'
            if s.count('0') == num_remove:
                return_list.append(list(map(lambda x: int(x), s)))
        random.shuffle(return_list)
        return_list = return_list[:num_mask]
            
    return return_list # type: ignore
            
if __name__ == '__main__':
    task = 'scratchpad' # choice in ['direct', 'scratchpad', 'rf']
    test_num = 14
    do_mask = True
    with open('test_sample_base.pkl', 'rb') as rf:
        import pickle
        test_samples = pickle.load(rf)
    for test_time in tqdm(range(test_num), total=test_num): 
        random.shuffle(test_samples)
        for test_sample in tqdm(test_samples):
            a, b = test_sample
            accu = test_a_b(a, b, type="direct", detail=False, test_num=10)
            if accu > 0.2:
                continue
            random_icl_samples = generate_random_icl_sample(num=5, a_length=len(str(a)), b_length=len(str(b)), remove_similar=True, orig_a=a, orig_b=b, similar_type='replace', similar_num_digit=1)
            similar_icl_samples = generate_similar_icl_sample(num=5, a=a, b=b, similar='replace', num_digit=1, sure_every_digit=True)
            icl_samples = random_icl_samples + similar_icl_samples
            random.shuffle(icl_samples)
            similar_idx = [i for i in range(len(icl_samples)) if icl_samples[i] in similar_icl_samples]
            
            icl_accu = test_a_b(a=a, b=b, few_shot=icl_samples, test_num=10, detail=False, type=task)
            improve = icl_accu - accu
            print(f'a: {a:,}, b: {b:,}, orig accu: {100*accu:.2f}, icl accu: {100*icl_accu:.2f}, improve: {100*improve:.2f}')
            if improve < 0.8:
                continue
            
            contribution = [0.0 for _ in icl_samples]
            counts = [0 for _ in icl_samples]
            if not do_mask:
                continue
            
            for mask in tqdm(sample_remove_index(num_example=len(icl_samples), num_mask=None, num_remove=None, similar_index=similar_idx)):
                # 1 means be kept in icl samples, 0 means being removed.
                masked_icl_samples = [icl_samples[i] for i in range(len(icl_samples)) if mask[i] == 1]
                masked_icl_accu = test_a_b(a=a, b=b, few_shot=masked_icl_samples, test_num=10, detail=False, type=task)
                masked_improve = masked_icl_accu - accu
                for j in range(len(icl_samples)):
                    if mask[j] == 0: # if removed, we can test the contribution by compare the accu improvement before and after its removal
                        # contribution[j] += (improve - masked_improve) / (improve * mask.count(0))
                        contribution[j] += (improve - masked_improve) / (improve)
                        counts[j] += 1
            
            contribution = [c / counts[i] if counts[i] > 0 else 0.0 for i, c in enumerate(contribution)]
            result_save_name = f'base_{task}'
            import os
            if not os.path.exists(result_save_name):
                os.makedirs(result_save_name, exist_ok=True)
            with open(f'{result_save_name}/intermediate_result_{test_time}.pkl', 'wb') as f:
                import pickle
                pickle.dump((similar_idx, contribution, counts, icl_samples), f)           
            break
            
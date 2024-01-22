import json
import os
import re


def extract_answer(rationale):
    answer = rationale.split("\n")[-1].split(",")[1]
    return answer

def count_c(a_digit, b_digit, c, mode):
    a_digit = int(a_digit)
    b_digit = int(b_digit)
    c = int(c)
    if mode == "add":
        if a_digit + b_digit + c>= 10:
            return 1
        else:
            return 0
    elif mode == "subtract":
        if a_digit - b_digit + c < 0:
            return -1
        else:
            return 0

def count_sub(a_digit, b_digit, c):
    r = a_digit - b_digit + c
    if r < 0:
        return str(10 + r)
    else:
        return str(r)

def gen_cot_rationale(a, b, mode="add"):
    '''
    return the cot rationale for the question
    '''
    a_digits = [digit for digit in str(a)]
    b_digits = [digit for digit in str(b)]
    rationale = ""
    answer = ""
    c = 0
    if mode == "add":
        gt = int(a) + int(b)
        for _ in range(len(str(int(a)+int(b)))+1):
            line = f"{''.join(a_digits)}+{''.join(b_digits)},{answer},C:{c}\n"
            rationale += line
            if a_digits and b_digits:
                answer = str(int(a_digits[-1]) + int(b_digits[-1]) + c)[-1] + answer
                c = count_c(a_digits[-1], b_digits[-1], c, mode)
                a_digits.pop()
                b_digits.pop()
            elif a_digits:
                answer = str(int(a_digits[-1]) + c)[-1] + answer
                c = count_c(a_digits[-1], 0, c, mode)
                a_digits.pop()
            elif b_digits:
                answer = str(int(b_digits[-1]) + c)[-1] + answer
                c = count_c(0, b_digits[-1], c, mode)
                b_digits.pop()
            else:
                if c:
                    answer = str(c) + answer
                c = 0
    elif mode == "subtract":
        gt = int(a) - int(b)
        for _ in range(max(len(str(a)), len(str(b)))+1):
            line = f"{''.join(a_digits)}-{''.join(b_digits)},{answer},C:{c}\n"
            rationale += line
            if a_digits and b_digits:
                answer = count_sub(int(a_digits[-1]),int(b_digits[-1]),c)[-1] + answer
                c = count_c(a_digits[-1], b_digits[-1], c, mode)
                a_digits.pop()
                b_digits.pop()
            elif a_digits:
                answer = count_sub(int(a_digits[-1]),0,c)[-1] + answer
                c = count_c(a_digits[-1], 0, c, mode)
                a_digits.pop()
            elif b_digits:
                answer = count_sub(0,int(b_digits[-1]),c)[-1] + answer
                c = count_c(0, b_digits[-1], c, mode)
                b_digits.pop()
            else:
                if c:
                    answer = count_sub(0,0,c) + answer
                c = 0
    rationale = rationale.strip()
    assert int(extract_answer(rationale)) == int(gt)
    return f"{rationale}\n{gt}"

def gen_mod_add_cot_rationale(a, b, P):
    rationale = gen_cot_rationale(a, b, mode="add")
    if a + b >= P:
        rationale += f"\n{a+b}>={P}\n{a+b}-{P}=\n"
        rationale += gen_cot_rationale(a+b, P, mode="subtract")
    else:
        rationale += f"\n{a+b}<{P}\n{a+b}"
    try:
        assert str((a+b)%P) == re.findall(r'[0-9]+\.?[0-9]*', rationale)[-1]
    except:
        print(rationale)
        print((a+b)%P)
        raise AssertionError()
    return rationale


if __name__ == "__main__":
    title = "1hole_(56, 56)_10_441_0-113"
    P = 113
    with open(f"{title}/train.json", "r") as f:
        train_samples = json.load(f)

    with open(f"{title}/test.json", "r") as f:
        test_samples = json.load(f)

    cot_train_samples = []
    for train_sample in train_samples:
        gt = train_sample['answers'][0]
        train_sample['answers'] = [gen_mod_add_cot_rationale(train_sample["a"], train_sample["b"], P)]
        train_sample["gt"] = gt
        cot_train_samples.append(train_sample)

    cot_test_samples = []
    for test_sample in test_samples:
        gt = test_sample['answers'][0]
        test_sample['answers'] = [gen_mod_add_cot_rationale(test_sample["a"], test_sample["b"], P)]
        test_sample["gt"] = gt
        cot_test_samples.append(test_sample)

    os.mkdir(f"{title}_cot")
    with open(f"{title}_cot/train.json", "w") as f:
        json.dump(cot_train_samples, f)

    with open(f"{title}_cot/test.json", "w") as f:
        json.dump(cot_test_samples, f)

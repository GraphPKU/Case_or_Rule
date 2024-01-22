import json
import os


def extract_answer(rationale):
    answer = rationale.split("\n")[-1].split(",")[1]
    return answer

def count_add(a_digit, b_digit, c, base):
    a_digit = int(a_digit)
    b_digit = int(b_digit)
    c = int(c)
    r = a_digit + b_digit + c
    if r >= base:
        c = 1
        r = r - base
    else:
        c = 0
    return r, c

def convert_to_base(n, base):
    s = ""
    n = int(n)
    while True:
        if n < base:
            s = str(n) + s
            break
        else:
            d = n % base
            n = n // base
            s = str(d) + s
    return s

def gen_cot_rationale(a, b, gt, base, cot="scratch"):
    '''
    return the cot rationale for the question
    '''
    if cot == "scratch":
        a_digits = [digit for digit in str(a)]
        b_digits = [digit for digit in str(b)]
        rationale = ""
        answer = ""
        c = 0
        for _ in range(len(convert_to_base(int(a)+int(b), base))+1):
            line = f"{''.join(a_digits)}+{''.join(b_digits)},{answer},C:{c}\n"
            rationale += line
            if a_digits and b_digits:
                r, c = count_add(a_digits[-1], b_digits[-1], c, base)
                answer = str(r) + answer
                a_digits.pop()
                b_digits.pop()
            elif a_digits:
                r, c = count_add(a_digits[-1], 0, c, base)
                answer = str(r) + answer
                a_digits.pop()
            elif b_digits:
                r, c = count_add(0, b_digits[-1], c, base)
                answer = str(r) + answer
                b_digits.pop()
            else:
                if c:
                    r, c = count_add(0, 0, c, base)
                    answer =  str(r) + answer
                c = 0
        rationale = rationale.strip()
        try:
            assert int(extract_answer(rationale)) == int(gt)
        except:
            print(rationale)
            print(gt)
            raise AssertionError
        return f"{rationale}\n{gt}"


if __name__ == "__main__":
    title = "1hole_(50, 50)_10_441_0-100"
    B = 9
    with open(f"{title}/train.json", "r") as f:
        train_samples = json.load(f)

    with open(f"{title}/test.json", "r") as f:
        test_samples = json.load(f)

    cot_train_samples = []
    for train_sample in train_samples:
        gt = train_sample['answers'][0]
        a = convert_to_base(train_sample["a"], B)
        b = convert_to_base(train_sample["b"], B)
        train_sample['answers'] = [gen_cot_rationale(a, b, gt, B)]
        train_sample["gt"] = gt
        cot_train_samples.append(train_sample)

    cot_test_samples = []
    for test_sample in test_samples:
        gt = test_sample['answers'][0]
        a = convert_to_base(test_sample["a"], B)
        b = convert_to_base(test_sample["b"], B)
        test_sample['answers'] = [gen_cot_rationale(a, b, gt, B)]
        test_sample["gt"] = gt
        cot_test_samples.append(test_sample)

    os.mkdir(f"{title}_cot")
    with open(f"{title}_cot/train.json", "w") as f:
        json.dump(cot_train_samples, f)

    with open(f"{title}_cot/test.json", "w") as f:
        json.dump(cot_test_samples, f)

import json


def extract_answer(rationale):
    answer = rationale.split("\n")[-1].split(",")[1]
    return answer

def count_c(a_digit, b_digit, c):
    a_digit = int(a_digit)
    b_digit = int(b_digit)
    c = int(c)
    if a_digit + b_digit + c>= 10:
        return 1
    else:
        return 0

def gen_cot_rationale(a, b, gt, cot="scratch"):
    '''
    return the cot rationale for the question
    '''
    if cot == "scratch":
        a_digits = [digit for digit in str(a)]
        b_digits = [digit for digit in str(b)]
        rationale = ""
        answer = ""
        c = 0
        for _ in range(len(str(int(a)+int(b)))+1):
            line = f"{''.join(a_digits)}+{''.join(b_digits)},{answer},C:{c}\n"
            rationale += line
            if a_digits and b_digits:
                answer = str(int(a_digits[-1]) + int(b_digits[-1]) + c)[-1] + answer
                c = count_c(a_digits[-1], b_digits[-1], c)
                a_digits.pop()
                b_digits.pop()
            elif a_digits:
                answer = str(int(a_digits[-1]) + c)[-1] + answer
                c = count_c(a_digits[-1], 0, c)
                a_digits.pop()
            elif b_digits:
                answer = str(int(b_digits[-1]) + c)[-1] + answer
                c = count_c(0, b_digits[-1], c)
                b_digits.pop()
            else:
                if c:
                    answer = str(c) + answer
                c = 0
        rationale = rationale.strip()
        assert int(extract_answer(rationale)) == int(gt)
        return f"{rationale}\n{gt}"


if __name__ == "__main__":
    task = "a100_b100_5hole"
    with open(f"{task}/train.json", "r") as f:
        train_samples = json.load(f)

    with open(f"{task}/test.json", "r") as f:
        test_samples = json.load(f)

    cot_train_samples = []
    for train_sample in train_samples:
        gt = train_sample['answers'][0]
        train_sample['answers'] = [gen_cot_rationale(train_sample["a"], train_sample["b"], gt)]
        train_sample["gt"] = gt
        cot_train_samples.append(train_sample)

    cot_test_samples = []
    for test_sample in test_samples:
        gt = test_sample['answers'][0]
        test_sample['answers'] = [gen_cot_rationale(test_sample["a"], test_sample["b"], gt)]
        test_sample["gt"] = gt
        cot_test_samples.append(test_sample)

    with open(f"{task}/train_cot.json", "w") as f:
        json.dump(cot_train_samples, f)

    with open(f"{task}/test_cot.json", "w") as f:
        json.dump(cot_test_samples, f)

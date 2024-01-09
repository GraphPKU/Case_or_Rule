import json

def update(file_path = "mod_add.json"):
    with open(file_path, "r") as f:
        dataset = json.load(f)
    new_dataset = []
    for i, data in enumerate(dataset):
        data["answers"] = [str(data["answer"])]
        new_dataset.append(data)
    with open(file_path, "w") as f:
        json.dump(new_dataset, f)

if __name__ == "__main__":
    update()
    update("mod_add_train.json")
    update("mod_add_test.json")

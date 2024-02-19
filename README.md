# Case-Based or Rule-Based: How Do Transformers Do the Math?

We explore whether LLMs perform case-based or rule-based reasoning in this work.

## requirements
Tested combination of python packages that can successfully complete the program is listed in [requirements.txt](/requirements.txt). You can run the following script to install them.

```bash
pip install -r requirements.txt
```

## replication of Leave-Square-Out

To replicate our main experiments of Leaving-Square-Out, you need to download the GPT-2 or GPT-2 Medium models and put them in `.\pretrained_models`. Then, you can run the script [train.py](/train.py) to fine-tune the pre-trained models.

## datasets

We provide the datasets for our main experiments in `.\datasets`. In each dataset, we provide a figure showing the train-test split `data_split.png`.

## Llama
We adopt the FastChat Framework to finetune Llama-7B in `./llama`.

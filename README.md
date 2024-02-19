# Case-Based or Rule-Based: How Do Transformers Do the Math?

We explore whether LLMs perform case-based or rule-based reasoning in this work.

## replication

To replicate our main experiments of Leaving-Square-Out, you need to download the GPT-2 or GPT-2 Medium models and put them in `.\pretrained_models`. Then, you can run the script [train.py](/train.py) to train the pre-trained models.

## datasets

We provide the datasets for our main experiments in `.\datasets`. In each dataset, we provide a figure showing the train-test split `data_split.png`.

## checkpoints

Due to the space limit, we are unable to provide our checkpoints here. We promise to provide them later in the github.

## Llama
We adopt the FastChat Framework to finetune Llama-7B in `./llama`.
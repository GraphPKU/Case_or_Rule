# Case-Based or Rule-Based: How Do Transformers Do the Math?

We explore whether LLMs perform case-based or rule-based reasoning in this work.

:star: Official code for [Case-Based or Rule-Based: How Do Transformers Do the Math?](https://arxiv.org/abs/2402.17709).

## Requirements
Tested combination of python packages that can successfully complete the program is listed in [requirements.txt](/requirements.txt). You can run the following script to install them.

```bash
pip install -r requirements.txt
```

## Replication of Leave-Square-Out

To replicate our main experiments of Leaving-Square-Out, you need to download the GPT-2 or GPT-2 Medium models and put them in `.\pretrained_models`. Then, you can run the script [train.py](/train.py) to fine-tune the pre-trained models.

## Datasets

We provide the datasets for our main experiments in `.\datasets`. In each dataset, we provide a figure showing the train-test split `data_split.png`.


## Llama
We adopt the FastChat Framework to finetune Llama-7B in `./llama`.

## Citation
If you want to use the code for your research, please cite our paper:
```
@misc{hu2024casebased,
      title={Case-Based or Rule-Based: How Do Transformers Do the Math?},
      author={Yi Hu and Xiaojuan Tang and Haotong Yang and Muhan Zhang},
      year={2024},
      eprint={2402.17709},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
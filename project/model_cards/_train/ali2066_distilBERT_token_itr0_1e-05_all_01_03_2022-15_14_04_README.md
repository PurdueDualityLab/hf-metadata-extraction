---
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- precision
- recall
- f1
- accuracy
model-index:
- name: distilBERT_token_itr0_1e-05_all_01_03_2022-15_14_04
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# distilBERT_token_itr0_1e-05_all_01_03_2022-15_14_04

This model is a fine-tuned version of [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert-base-uncased-finetuned-sst-2-english) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.3121
- Precision: 0.1204
- Recall: 0.2430
- F1: 0.1611
- Accuracy: 0.8538

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss | Precision | Recall | F1     | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:---------:|:------:|:------:|:--------:|
| No log        | 1.0   | 30   | 0.4480          | 0.0209    | 0.0223 | 0.0216 | 0.7794   |
| No log        | 2.0   | 60   | 0.3521          | 0.0559    | 0.1218 | 0.0767 | 0.8267   |
| No log        | 3.0   | 90   | 0.3177          | 0.1208    | 0.2504 | 0.1629 | 0.8487   |
| No log        | 4.0   | 120  | 0.3009          | 0.1296    | 0.2607 | 0.1731 | 0.8602   |
| No log        | 5.0   | 150  | 0.2988          | 0.1393    | 0.2693 | 0.1836 | 0.8599   |


### Framework versions

- Transformers 4.15.0
- Pytorch 1.10.1+cu113
- Datasets 1.18.0
- Tokenizers 0.10.3
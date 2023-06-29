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
- name: bert-base-uncased_token_itr0_0.0001_all_01_03_2022-04_48_27
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bert-base-uncased_token_itr0_0.0001_all_01_03_2022-04_48_27

This model is a fine-tuned version of [bert-base-uncased](https://huggingface.co/bert-base-uncased) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2899
- Precision: 0.3170
- Recall: 0.5261
- F1: 0.3956
- Accuracy: 0.8799

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 32
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss | Precision | Recall | F1     | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:---------:|:------:|:------:|:--------:|
| No log        | 1.0   | 30   | 0.2912          | 0.2752    | 0.4444 | 0.3400 | 0.8730   |
| No log        | 2.0   | 60   | 0.2772          | 0.4005    | 0.4589 | 0.4277 | 0.8911   |
| No log        | 3.0   | 90   | 0.2267          | 0.3642    | 0.5281 | 0.4311 | 0.9043   |
| No log        | 4.0   | 120  | 0.2129          | 0.3617    | 0.5455 | 0.4350 | 0.9140   |
| No log        | 5.0   | 150  | 0.2399          | 0.3797    | 0.5556 | 0.4511 | 0.9114   |


### Framework versions

- Transformers 4.15.0
- Pytorch 1.10.1+cu113
- Datasets 1.18.0
- Tokenizers 0.10.3
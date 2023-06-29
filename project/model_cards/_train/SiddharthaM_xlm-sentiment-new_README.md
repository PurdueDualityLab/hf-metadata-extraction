---
license: mit
tags:
- generated_from_trainer
metrics:
- accuracy
- precision
- recall
- f1
model-index:
- name: xlm-sentiment-new
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# xlm-sentiment-new

This model is a fine-tuned version of [xlm-roberta-base](https://huggingface.co/xlm-roberta-base) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6166
- Accuracy: 0.7405
- Precision: 0.7375
- Recall: 0.7405
- F1: 0.7386

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
- num_epochs: 6

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy | Precision | Recall | F1     |
|:-------------:|:-----:|:----:|:---------------:|:--------:|:---------:|:------:|:------:|
| No log        | 1.0   | 296  | 0.5519          | 0.7310   | 0.7266    | 0.7310 | 0.7277 |
| 0.5719        | 2.0   | 592  | 0.5569          | 0.75     | 0.7562    | 0.75   | 0.7302 |
| 0.5719        | 3.0   | 888  | 0.5320          | 0.7243   | 0.7269    | 0.7243 | 0.7254 |
| 0.477         | 4.0   | 1184 | 0.5771          | 0.7300   | 0.7264    | 0.7300 | 0.7276 |
| 0.477         | 5.0   | 1480 | 0.6051          | 0.7376   | 0.7361    | 0.7376 | 0.7368 |
| 0.428         | 6.0   | 1776 | 0.6166          | 0.7405   | 0.7375    | 0.7405 | 0.7386 |


### Framework versions

- Transformers 4.24.0.dev0
- Pytorch 1.11.0+cu102
- Datasets 2.6.1
- Tokenizers 0.13.1
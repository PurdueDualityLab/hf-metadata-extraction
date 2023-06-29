---
license: mit
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: test_trainer
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# test_trainer

This model is a fine-tuned version of [cointegrated/rubert-tiny](https://huggingface.co/cointegrated/rubert-tiny) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.7773
- Accuracy: 0.6375

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 1

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 0.7753        | 1.0   | 400  | 0.7773          | 0.6375   |


### Framework versions

- Transformers 4.18.0
- Pytorch 1.11.0+cu113
- Datasets 2.3.2
- Tokenizers 0.12.1
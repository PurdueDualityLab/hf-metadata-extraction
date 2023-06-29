---
license: apache-2.0
tags:
- translation
- generated_from_trainer
model-index:
- name: nils-nl-to-rx-pt-v7
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# nils-nl-to-rx-pt-v7

This model is a fine-tuned version of [Helsinki-NLP/opus-mt-de-en](https://huggingface.co/Helsinki-NLP/opus-mt-de-en) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0224

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.4389        | 1.0   | 500  | 0.0470          |
| 0.0533        | 2.0   | 1000 | 0.0286          |
| 0.0346        | 3.0   | 1500 | 0.0224          |


### Framework versions

- Transformers 4.24.0
- Pytorch 1.12.1+cu113
- Datasets 2.6.1
- Tokenizers 0.13.1
---
tags:
- image-classification
- timm
- generated_from_trainer
library_tag: timm
datasets:
- cats_vs_dogs
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# my-cool-timm-model-2

This model is a fine-tuned version of [resnet18](https://huggingface.co/resnet18) on the cats_vs_dogs dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2510
- Acc1: 95.2150
- Acc5: 100.0

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
- train_batch_size: 256
- eval_batch_size: 256
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- training_steps: 10
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Acc1    | Acc5  |
|:-------------:|:-----:|:----:|:---------------:|:-------:|:-----:|
| No log        | 0.07  | 5    | 0.3436          | 92.0820 | 100.0 |
| 0.4914        | 0.14  | 10   | 0.2510          | 95.2150 | 100.0 |


### Framework versions

- Transformers 4.12.3
- Pytorch 1.10.0+cu111
- Datasets 1.15.1
- Tokenizers 0.10.3
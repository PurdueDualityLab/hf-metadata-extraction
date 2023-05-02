---
license: apache-2.0
tags:
- generated_from_trainer
model-index:
- name: chinese-bert-wwm-finetuned-product-1
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# chinese-bert-wwm-finetuned-product-1

This model is a fine-tuned version of [hfl/chinese-bert-wwm](https://huggingface.co/hfl/chinese-bert-wwm) on the None dataset.
It achieves the following results on the evaluation set:
- eval_loss: 0.0000
- eval_runtime: 10.6737
- eval_samples_per_second: 362.572
- eval_steps_per_second: 5.715
- epoch: 11.61
- step: 18797

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
- train_batch_size: 256
- eval_batch_size: 64
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 20

### Framework versions

- Transformers 4.17.0
- Pytorch 1.6.0
- Datasets 2.0.0
- Tokenizers 0.11.6

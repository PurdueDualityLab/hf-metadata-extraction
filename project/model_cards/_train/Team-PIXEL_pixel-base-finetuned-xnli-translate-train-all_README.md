---
language:
- en
- ar
- bg
- de
- el
- fr
- hi
- ru
- es
- sw
- th
- tr
- ur
- vi
- zh
tags:
- generated_from_trainer
datasets:
- xnli
metrics:
- accuracy
model-index:
- name: pixel-base-finetuned-xnli-translate-train-all
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: XNLI
      type: xnli
      args: xnli
    metrics:
    - name: Joint validation accuracy
      type: accuracy
      value: 0.6254886211512718
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# pixel-base-finetuned-xnli-translate-train-all

This model is a fine-tuned version of [Team-PIXEL/pixel-base](https://huggingface.co/Team-PIXEL/pixel-base) on the XNLI dataset.

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
- eval_batch_size: 8
- seed: 555
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 1000
- training_steps: 50000
- mixed_precision_training: Apex, opt level O1

### Framework versions

- Transformers 4.17.0
- Pytorch 1.11.0
- Datasets 2.0.0
- Tokenizers 0.12.1
---
language:
- en
- ro
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- wmt16
metrics:
- bleu
model-index:
- name: finetuned-mt5-base
  results:
  - task:
      name: Translation
      type: translation
    dataset:
      name: wmt16 ro-en
      type: wmt16
      args: ro-en
    metrics:
    - name: Bleu
      type: bleu
      value: 27.1659
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# finetuned-mt5-base

This model is a fine-tuned version of [google/mt5-base](https://huggingface.co/google/mt5-base) on the wmt16 ro-en dataset.
It achieves the following results on the evaluation set:
- Loss: 1.3594
- Bleu: 27.1659
- Gen Len: 43.9575

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
- train_batch_size: 4
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 10.0

### Training results



### Framework versions

- Transformers 4.20.1
- Pytorch 1.12.0+cu102
- Datasets 2.3.2
- Tokenizers 0.12.1
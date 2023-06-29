---
language:
- ar
license: apache-2.0
tags:
- hf-ast-leaderboard
- generated_from_trainer
metrics:
- wer
model-index:
- name: Whisper Small arb - GP
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Whisper Small arb - GP

This model is a fine-tuned version of [openai/whisper-small](https://huggingface.co/openai/whisper-small) on the Dialect Arabic dataset.
It achieves the following results on the evaluation set:
- Loss: 2.1489
- Wer: 110.7984

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
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 4000
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Wer      |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 1.9933        | 1.89  | 1000 | 2.0970          | 125.2555 |
| 1.3119        | 3.79  | 2000 | 1.9818          | 113.1290 |
| 0.7643        | 5.68  | 3000 | 2.0559          | 115.4176 |
| 0.5144        | 7.58  | 4000 | 2.1489          | 110.7984 |


### Framework versions

- Transformers 4.26.0.dev0
- Pytorch 1.11.0
- Datasets 2.1.0
- Tokenizers 0.12.1
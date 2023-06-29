---
tags:
- generated_from_trainer
datasets:
- common_voice
model-index:
- name: facebook_large_CV_bn3
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# facebook_large_CV_bn3

This model is a fine-tuned version of [Sameen53/facebook_large_CV_bn](https://huggingface.co/Sameen53/facebook_large_CV_bn) on the common_voice dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2308
- Wer: 0.2379

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
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 200
- num_epochs: 6
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Wer    |
|:-------------:|:-----:|:----:|:---------------:|:------:|
| No log        | 0.87  | 1000 | 0.2473          | 0.2524 |
| 0.2308        | 1.73  | 2000 | 0.2073          | 0.2450 |
| 0.261         | 2.6   | 3000 | 0.2036          | 0.2345 |
| 0.2498        | 3.47  | 4000 | 0.1916          | 0.2311 |
| 0.2433        | 4.33  | 5000 | 0.1869          | 0.2344 |
| 0.2588        | 5.2   | 6000 | 0.2308          | 0.2379 |


### Framework versions

- Transformers 4.20.1
- Pytorch 1.11.0
- Datasets 2.1.0
- Tokenizers 0.12.1
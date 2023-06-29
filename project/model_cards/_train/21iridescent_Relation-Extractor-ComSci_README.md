---
license: cc-by-nc-sa-4.0
tags:
- generated_from_trainer
model-index:
- name: output
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# output

This model is a fine-tuned version of [Babelscape/rebel-large](https://huggingface.co/Babelscape/rebel-large) on an unknown dataset.

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
- eval_batch_size: 16
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 1

### Training results

| Training Loss | Epoch | Step | Validation Loss | Precision | Recall | F1-measure |
|:-------------:|:-----:|:----:|:---------------:|:---------:|:------:|:----------:|
| No log        | 1.0   | 236  | 0.3225          | 0.8889    | 0.8889 | 0.8889     |


### Framework versions

- Transformers 4.21.1
- Pytorch 1.9.0+cu111
- Datasets 2.4.0
- Tokenizers 0.12.1
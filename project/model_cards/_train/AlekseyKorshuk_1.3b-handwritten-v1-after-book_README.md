---
tags:
- generated_from_trainer
datasets:
- AlekseyKorshuk/dalio-handwritten-io
metrics:
- accuracy
model-index:
- name: 1.3b-handwritten-v1-after-book
  results:
  - task:
      name: Causal Language Modeling
      type: text-generation
    dataset:
      name: AlekseyKorshuk/dalio-handwritten-io
      type: AlekseyKorshuk/dalio-handwritten-io
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.06691769057999736
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# 1.3b-handwritten-v1-after-book

This model is a fine-tuned version of [/models/1.3b-dalio-principles-book](https://huggingface.co//models/1.3b-dalio-principles-book) on the AlekseyKorshuk/dalio-handwritten-io dataset.
It achieves the following results on the evaluation set:
- Loss: 2.0566
- Accuracy: 0.0669

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 3e-05
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- distributed_type: multi-GPU
- num_devices: 8
- total_train_batch_size: 32
- total_eval_batch_size: 32
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- num_epochs: 3.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 2.3721        | 0.2   | 1    | 2.2148          | 0.0641   |
| 2.241         | 0.4   | 2    | 2.2148          | 0.0641   |
| 2.469         | 0.6   | 3    | 2.1348          | 0.0653   |
| 2.3735        | 0.8   | 4    | 2.1309          | 0.0648   |
| 2.2755        | 1.0   | 5    | 2.1133          | 0.0652   |
| 2.0428        | 1.2   | 6    | 2.0938          | 0.0659   |
| 1.764         | 1.4   | 7    | 2.0781          | 0.0659   |
| 1.7458        | 1.6   | 8    | 2.0781          | 0.0661   |
| 1.868         | 1.8   | 9    | 2.0820          | 0.0660   |
| 1.9548        | 2.0   | 10   | 2.0703          | 0.0663   |
| 1.6772        | 2.2   | 11   | 2.0605          | 0.0665   |
| 1.3997        | 2.4   | 12   | 2.0566          | 0.0668   |
| 1.3717        | 2.6   | 13   | 2.0547          | 0.0669   |
| 1.5284        | 2.8   | 14   | 2.0547          | 0.0667   |
| 1.2264        | 3.0   | 15   | 2.0566          | 0.0669   |


### Framework versions

- Transformers 4.25.0.dev0
- Pytorch 1.12.1+cu113
- Datasets 2.3.2
- Tokenizers 0.12.1
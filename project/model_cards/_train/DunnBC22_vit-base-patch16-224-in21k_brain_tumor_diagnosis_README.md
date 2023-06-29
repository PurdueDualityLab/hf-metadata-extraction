---
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- imagefolder
metrics:
- accuracy
- f1
- recall
- precision
model-index:
- name: vit-base-patch16-224-in21k_brain_tumor_diagnosis
  results:
  - task:
      name: Image Classification
      type: image-classification
    dataset:
      name: imagefolder
      type: imagefolder
      config: default
      split: train
      args: default
    metrics:
    - name: Accuracy
      type: accuracy
      value: 0.9215686274509803
    - name: F1
      type: f1
      value: 0.9375
    - name: Recall
      type: recall
      value: 1.0
    - name: Precision
      type: precision
      value: 0.8823529411764706
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# vit-base-patch16-224-in21k_brain_tumor_diagnosis

This model is a fine-tuned version of [google/vit-base-patch16-224-in21k](https://huggingface.co/google/vit-base-patch16-224-in21k) on the imagefolder dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2591
- Accuracy: 0.9216
- F1: 0.9375
- Recall: 1.0
- Precision: 0.8824

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0002
- train_batch_size: 16
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 5

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy | F1     | Recall | Precision |
|:-------------:|:-----:|:----:|:---------------:|:--------:|:------:|:------:|:---------:|
| 0.7101        | 1.0   | 13   | 0.3351          | 0.9412   | 0.9474 | 0.9    | 1.0       |
| 0.7101        | 2.0   | 26   | 0.3078          | 0.9020   | 0.9231 | 1.0    | 0.8571    |
| 0.7101        | 3.0   | 39   | 0.2591          | 0.9216   | 0.9375 | 1.0    | 0.8824    |
| 0.7101        | 4.0   | 52   | 0.2702          | 0.9020   | 0.9123 | 0.8667 | 0.9630    |
| 0.7101        | 5.0   | 65   | 0.2855          | 0.9020   | 0.9123 | 0.8667 | 0.9630    |


### Framework versions

- Transformers 4.25.1
- Pytorch 1.12.1
- Datasets 2.8.0
- Tokenizers 0.12.1
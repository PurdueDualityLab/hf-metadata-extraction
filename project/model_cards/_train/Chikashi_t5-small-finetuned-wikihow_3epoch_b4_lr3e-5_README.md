---
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- wikihow
metrics:
- rouge
model-index:
- name: t5-small-finetuned-wikihow_3epoch_b4_lr3e-5
  results:
  - task:
      name: Sequence-to-sequence Language Modeling
      type: text2text-generation
    dataset:
      name: wikihow
      type: wikihow
      args: all
    metrics:
    - name: Rouge1
      type: rouge
      value: 26.1071
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# t5-small-finetuned-wikihow_3epoch_b4_lr3e-5

This model is a fine-tuned version of [t5-small](https://huggingface.co/t5-small) on the wikihow dataset.
It achieves the following results on the evaluation set:
- Loss: 2.4351
- Rouge1: 26.1071
- Rouge2: 9.3627
- Rougel: 22.0825
- Rougelsum: 25.4514
- Gen Len: 18.474

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
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step   | Validation Loss | Rouge1  | Rouge2 | Rougel  | Rougelsum | Gen Len |
|:-------------:|:-----:|:------:|:---------------:|:-------:|:------:|:-------:|:---------:|:-------:|
| 2.9216        | 0.13  | 5000   | 2.6385          | 23.8039 | 7.8863 | 20.0109 | 23.0802   | 18.3481 |
| 2.8158        | 0.25  | 10000  | 2.5884          | 24.2567 | 8.2003 | 20.438  | 23.5325   | 18.3833 |
| 2.7743        | 0.38  | 15000  | 2.5623          | 24.8471 | 8.3768 | 20.8711 | 24.1114   | 18.2901 |
| 2.7598        | 0.51  | 20000  | 2.5368          | 25.1566 | 8.6721 | 21.1896 | 24.4558   | 18.3561 |
| 2.7192        | 0.64  | 25000  | 2.5220          | 25.3477 | 8.8106 | 21.3799 | 24.6742   | 18.3108 |
| 2.7207        | 0.76  | 30000  | 2.5114          | 25.5912 | 8.998  | 21.5508 | 24.9344   | 18.3445 |
| 2.7041        | 0.89  | 35000  | 2.4993          | 25.457  | 8.8644 | 21.4516 | 24.7965   | 18.4354 |
| 2.687         | 1.02  | 40000  | 2.4879          | 25.5886 | 8.9766 | 21.6794 | 24.9512   | 18.4035 |
| 2.6652        | 1.14  | 45000  | 2.4848          | 25.7367 | 9.078  | 21.7096 | 25.0924   | 18.4328 |
| 2.6536        | 1.27  | 50000  | 2.4761          | 25.7368 | 9.1609 | 21.729  | 25.0866   | 18.3117 |
| 2.6589        | 1.4   | 55000  | 2.4702          | 25.7738 | 9.1413 | 21.7492 | 25.114    | 18.4862 |
| 2.6384        | 1.53  | 60000  | 2.4620          | 25.7433 | 9.1356 | 21.8198 | 25.0896   | 18.489  |
| 2.6337        | 1.65  | 65000  | 2.4595          | 26.0919 | 9.2605 | 21.9447 | 25.4065   | 18.4083 |
| 2.6375        | 1.78  | 70000  | 2.4557          | 26.0912 | 9.3469 | 22.0182 | 25.4428   | 18.4133 |
| 2.6441        | 1.91  | 75000  | 2.4502          | 26.1366 | 9.3143 | 22.058  | 25.4673   | 18.4972 |
| 2.6276        | 2.03  | 80000  | 2.4478          | 25.9929 | 9.2464 | 21.9271 | 25.3263   | 18.469  |
| 2.6062        | 2.16  | 85000  | 2.4467          | 26.0465 | 9.3166 | 22.0342 | 25.3998   | 18.3777 |
| 2.6126        | 2.29  | 90000  | 2.4407          | 26.1953 | 9.3848 | 22.1148 | 25.5161   | 18.467  |
| 2.6182        | 2.42  | 95000  | 2.4397          | 26.1331 | 9.3626 | 22.1076 | 25.4627   | 18.4413 |
| 2.6041        | 2.54  | 100000 | 2.4375          | 26.1301 | 9.3567 | 22.0869 | 25.465    | 18.4929 |
| 2.5996        | 2.67  | 105000 | 2.4367          | 26.0956 | 9.3314 | 22.063  | 25.4242   | 18.5074 |
| 2.6144        | 2.8   | 110000 | 2.4355          | 26.1764 | 9.4157 | 22.1231 | 25.5175   | 18.4729 |
| 2.608         | 2.93  | 115000 | 2.4351          | 26.1071 | 9.3627 | 22.0825 | 25.4514   | 18.474  |


### Framework versions

- Transformers 4.18.0
- Pytorch 1.10.0+cu111
- Datasets 2.0.0
- Tokenizers 0.11.6
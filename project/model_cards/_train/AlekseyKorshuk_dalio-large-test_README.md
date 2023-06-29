---
license: other
tags:
- generated_from_trainer
datasets:
- AlekseyKorshuk/dalio-handwritten-io
metrics:
- accuracy
model-index:
- name: dalio-large-test
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
      value: 0.047694543532831285
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# dalio-large-test

This model is a fine-tuned version of [facebook/opt-125m](https://huggingface.co/facebook/opt-125m) on the AlekseyKorshuk/dalio-handwritten-io dataset.
It achieves the following results on the evaluation set:
- Loss: 3.1016
- Accuracy: 0.0477

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
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- distributed_type: multi-GPU
- num_devices: 8
- total_train_batch_size: 8
- total_eval_batch_size: 8
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: cosine
- training_steps: 1

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 3.3342        | 0.05  | 1    | 3.1016          | 0.0477   |


### Framework versions

- Transformers 4.25.0.dev0
- Pytorch 1.12.1+cu113
- Datasets 2.3.2
- Tokenizers 0.12.1
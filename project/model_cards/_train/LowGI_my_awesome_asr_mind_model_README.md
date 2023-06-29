---
license: apache-2.0
tags:
- generated_from_trainer
datasets:
- minds14
metrics:
- wer
model-index:
- name: my_awesome_asr_mind_model
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: minds14
      type: minds14
      config: en-US
      split: train[:100]
      args: en-US
    metrics:
    - name: Wer
      type: wer
      value: 1.0
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# my_awesome_asr_mind_model

This model is a fine-tuned version of [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) on the minds14 dataset.
It achieves the following results on the evaluation set:
- Loss: 61.8991
- Wer: 1.0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.1
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 1
- training_steps: 1

### Training results

| Training Loss | Epoch | Step | Validation Loss | Wer |
|:-------------:|:-----:|:----:|:---------------:|:---:|
| 47.4031       | 0.2   | 1    | 61.8991         | 1.0 |


### Framework versions

- Transformers 4.24.0
- Pytorch 1.13.0+cpu
- Datasets 2.7.1
- Tokenizers 0.13.2
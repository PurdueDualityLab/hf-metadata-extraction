---
language:
- zh
license: apache-2.0
tags:
- whisper-event
- generated_from_trainer
datasets:
- mozilla-foundation/common_voice_11_0
metrics:
- wer
model-index:
- name: 'Whisper Small Chinese (Taiwan) '
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: mozilla-foundation/common_voice_11_0 zh-TW
      type: mozilla-foundation/common_voice_11_0
      config: zh-TW
      split: test
      args: zh-TW
    metrics:
    - name: Wer
      type: wer
      value: 41.96519959058342
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# Whisper Small Chinese (Taiwan) 

This model is a fine-tuned version of [openai/whisper-small](https://huggingface.co/openai/whisper-small) on the mozilla-foundation/common_voice_11_0 zh-TW dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2283
- Wer: 41.9652

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
- train_batch_size: 64
- eval_batch_size: 32
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 5000
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Wer     |
|:-------------:|:-----:|:----:|:---------------:|:-------:|
| 0.0049        | 6.02  | 1000 | 0.2283          | 41.9652 |
| 0.0008        | 13.02 | 2000 | 0.2556          | 42.0266 |
| 0.0004        | 20.01 | 3000 | 0.2690          | 42.4156 |
| 0.0003        | 27.0  | 4000 | 0.2788          | 42.7840 |
| 0.0002        | 33.02 | 5000 | 0.2826          | 43.0297 |


### Framework versions

- Transformers 4.26.0.dev0
- Pytorch 1.13.1+cu117
- Datasets 2.7.1.dev0
- Tokenizers 0.13.2
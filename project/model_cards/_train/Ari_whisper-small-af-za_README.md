---
language:
- af
license: apache-2.0
tags:
- whisper-event
- generated_from_trainer
- hf-asr-leaderboard

datasets:
- google/fleurs
- openslr/SLR32
model-index:
- name: whisper-small-af-za - Ari
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    metrics:
    - name: Wer
      type: wer
      value: 0.0
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# whisper-small-af-za - Ari

This model is a fine-tuned version of [openai/whisper-small](https://huggingface.co/openai/whisper-small) on the Common Voice 11.0 dataset.
It achieves the following results on the evaluation set:
- eval_loss: 0.0002
- eval_wer: 0.0
- eval_runtime: 77.0592
- eval_samples_per_second: 2.569
- eval_steps_per_second: 0.324
- epoch: 14.6
- step: 2000

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
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- training_steps: 5000
- mixed_precision_training: Native AMP

### Framework versions

- Transformers 4.26.0.dev0
- Pytorch 1.13.0+cu116
- Datasets 2.7.1.dev0
- Tokenizers 0.13.2

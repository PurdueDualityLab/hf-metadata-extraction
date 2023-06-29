---
language:
- fr
license: apache-2.0
tags:
- automatic-speech-recognition
- mozilla-foundation/common_voice_7_0
- generated_from_trainer
- robust-speech-event
- hf-asr-leaderboard
datasets:
- mozilla-foundation/common_voice_7_0
model-index:
- name: XLS-R-300M - French
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Common Voice 7
      type: mozilla-foundation/common_voice_7_0
      args: fr
    metrics:
    - name: Test WER
      type: wer
      value: 24.56
    - name: Test CER
      type: cer
      value: 7.3
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Robust Speech Event - Dev Data
      type: speech-recognition-community-v2/dev_data
      args: fr
    metrics:
    - name: Test WER
      type: wer
      value: 63.62
    - name: Test CER
      type: cer
      value: 17.2
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Robust Speech Event - Test Data
      type: speech-recognition-community-v2/eval_data
      args: fr
    metrics:
    - name: Test WER
      type: wer
      value: 66.45
---
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

## Model description

This model is a fine-tuned version of [facebook/wav2vec2-xls-r-300m](https://huggingface.co/facebook/wav2vec2-xls-r-300m) on the MOZILLA-FOUNDATION/COMMON_VOICE_7_0 - FR dataset.

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 7.5e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- gradient_accumulation_steps: 8
- total_train_batch_size: 128
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 2000
- num_epochs: 2.0
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Wer    |
|:-------------:|:-----:|:----:|:---------------:|:------:|
| 3.495         | 0.16  | 500  | 3.3883          | 1.0    |
| 2.9095        | 0.32  | 1000 | 2.9152          | 1.0000 |
| 1.8434        | 0.49  | 1500 | 1.0473          | 0.7446 |
| 1.4298        | 0.65  | 2000 | 0.5729          | 0.5130 |
| 1.1937        | 0.81  | 2500 | 0.3795          | 0.3450 |
| 1.1248        | 0.97  | 3000 | 0.3321          | 0.3052 |
| 1.0835        | 1.13  | 3500 | 0.3038          | 0.2805 |
| 1.0479        | 1.3   | 4000 | 0.2910          | 0.2689 |
| 1.0413        | 1.46  | 4500 | 0.2798          | 0.2593 |
| 1.014         | 1.62  | 5000 | 0.2727          | 0.2512 |
| 1.004         | 1.78  | 5500 | 0.2646          | 0.2471 |
| 0.9949        | 1.94  | 6000 | 0.2619          | 0.2457 |

It achieves the best result on STEP 6000 on the validation set:
- Loss: 0.2619
- Wer: 0.2457

### Framework versions

- Transformers 4.17.0.dev0
- Pytorch 1.10.2+cu102
- Datasets 1.18.2.dev0
- Tokenizers 0.11.0

### Evaluation Commands
1. To evaluate on `mozilla-foundation/common_voice_7` with split `test`

```bash
python eval.py --model_id Plim/xls-r-300m-fr --dataset mozilla-foundation/common_voice_7_0 --config fr --split test
```

2. To evaluate on `speech-recognition-community-v2/dev_data`

```bash
python eval.py --model_id Plim/xls-r-300m-fr --dataset speech-recognition-community-v2/dev_data --config fr --split validation --chunk_length_s 5.0 --stride_length_s 1.0
```
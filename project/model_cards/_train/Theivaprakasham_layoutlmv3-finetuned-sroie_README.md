---
tags:
- generated_from_trainer
datasets:
- sroie
metrics:
- precision
- recall
- f1
- accuracy
model-index:
- name: layoutlmv3-finetuned-sroie
  results:
  - task:
      name: Token Classification
      type: token-classification
    dataset:
      name: sroie
      type: sroie
      args: sroie
    metrics:
    - name: Precision
      type: precision
      value: 0.9370529327610873
    - name: Recall
      type: recall
      value: 0.9438040345821326
    - name: F1
      type: f1
      value: 0.9404163675520459
    - name: Accuracy
      type: accuracy
      value: 0.9945347083116948
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# layoutlmv3-finetuned-sroie

This model is a fine-tuned version of [microsoft/layoutlmv3-base](https://huggingface.co/microsoft/layoutlmv3-base) on the sroie dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0426
- Precision: 0.9371
- Recall: 0.9438
- F1: 0.9404
- Accuracy: 0.9945

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
- train_batch_size: 2
- eval_batch_size: 2
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- training_steps: 5000

### Training results

| Training Loss | Epoch | Step | Validation Loss | Precision | Recall | F1     | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:---------:|:------:|:------:|:--------:|
| No log        | 0.32  | 100  | 0.1127          | 0.6466    | 0.6102 | 0.6279 | 0.9729   |
| No log        | 0.64  | 200  | 0.0663          | 0.8215    | 0.7428 | 0.7802 | 0.9821   |
| No log        | 0.96  | 300  | 0.0563          | 0.8051    | 0.8718 | 0.8371 | 0.9855   |
| No log        | 1.28  | 400  | 0.0470          | 0.8766    | 0.8595 | 0.8680 | 0.9895   |
| 0.1328        | 1.6   | 500  | 0.0419          | 0.8613    | 0.9128 | 0.8863 | 0.9906   |
| 0.1328        | 1.92  | 600  | 0.0338          | 0.8888    | 0.9099 | 0.8993 | 0.9926   |
| 0.1328        | 2.24  | 700  | 0.0320          | 0.8690    | 0.9467 | 0.9062 | 0.9929   |
| 0.1328        | 2.56  | 800  | 0.0348          | 0.8960    | 0.9438 | 0.9193 | 0.9931   |
| 0.1328        | 2.88  | 900  | 0.0300          | 0.9169    | 0.9460 | 0.9312 | 0.9942   |
| 0.029         | 3.19  | 1000 | 0.0281          | 0.9080    | 0.9452 | 0.9262 | 0.9942   |
| 0.029         | 3.51  | 1100 | 0.0259          | 0.9174    | 0.9438 | 0.9304 | 0.9945   |
| 0.029         | 3.83  | 1200 | 0.0309          | 0.9207    | 0.9532 | 0.9366 | 0.9944   |
| 0.029         | 4.15  | 1300 | 0.0366          | 0.9195    | 0.9388 | 0.9291 | 0.9940   |
| 0.029         | 4.47  | 1400 | 0.0302          | 0.9343    | 0.9424 | 0.9383 | 0.9949   |
| 0.0174        | 4.79  | 1500 | 0.0349          | 0.9142    | 0.9517 | 0.9326 | 0.9939   |
| 0.0174        | 5.11  | 1600 | 0.0327          | 0.9322    | 0.9510 | 0.9415 | 0.9950   |
| 0.0174        | 5.43  | 1700 | 0.0317          | 0.9215    | 0.9561 | 0.9385 | 0.9938   |
| 0.0174        | 5.75  | 1800 | 0.0385          | 0.9282    | 0.9316 | 0.9299 | 0.9940   |
| 0.0174        | 6.07  | 1900 | 0.0342          | 0.9235    | 0.9481 | 0.9357 | 0.9944   |
| 0.0117        | 6.39  | 2000 | 0.0344          | 0.9287    | 0.9474 | 0.9379 | 0.9944   |
| 0.0117        | 6.71  | 2100 | 0.0388          | 0.9232    | 0.9445 | 0.9338 | 0.9941   |
| 0.0117        | 7.03  | 2200 | 0.0325          | 0.9269    | 0.9496 | 0.9381 | 0.9949   |
| 0.0117        | 7.35  | 2300 | 0.0343          | 0.9225    | 0.9438 | 0.9330 | 0.9941   |
| 0.0117        | 7.67  | 2400 | 0.0372          | 0.9216    | 0.9481 | 0.9347 | 0.9944   |
| 0.0081        | 7.99  | 2500 | 0.0385          | 0.9192    | 0.9589 | 0.9386 | 0.9944   |
| 0.0081        | 8.31  | 2600 | 0.0376          | 0.9293    | 0.9467 | 0.9379 | 0.9944   |
| 0.0081        | 8.63  | 2700 | 0.0425          | 0.9261    | 0.9474 | 0.9366 | 0.9941   |
| 0.0081        | 8.95  | 2800 | 0.0407          | 0.9266    | 0.9452 | 0.9358 | 0.9941   |
| 0.0081        | 9.27  | 2900 | 0.0403          | 0.9280    | 0.9467 | 0.9372 | 0.9941   |
| 0.0055        | 9.58  | 3000 | 0.0364          | 0.9287    | 0.9474 | 0.9379 | 0.9948   |
| 0.0055        | 9.9   | 3100 | 0.0427          | 0.9122    | 0.9510 | 0.9312 | 0.9941   |
| 0.0055        | 10.22 | 3200 | 0.0394          | 0.9223    | 0.9488 | 0.9354 | 0.9943   |
| 0.0055        | 10.54 | 3300 | 0.0393          | 0.9247    | 0.9561 | 0.9401 | 0.9945   |
| 0.0055        | 10.86 | 3400 | 0.0413          | 0.9334    | 0.9496 | 0.9414 | 0.9945   |
| 0.0049        | 11.18 | 3500 | 0.0400          | 0.9290    | 0.9517 | 0.9402 | 0.9945   |
| 0.0049        | 11.5  | 3600 | 0.0412          | 0.9317    | 0.9539 | 0.9427 | 0.9945   |
| 0.0049        | 11.82 | 3700 | 0.0419          | 0.9314    | 0.9481 | 0.9397 | 0.9947   |
| 0.0049        | 12.14 | 3800 | 0.0452          | 0.9243    | 0.9503 | 0.9371 | 0.9941   |
| 0.0049        | 12.46 | 3900 | 0.0412          | 0.9334    | 0.9496 | 0.9414 | 0.9947   |
| 0.0039        | 12.78 | 4000 | 0.0438          | 0.9294    | 0.9481 | 0.9387 | 0.9941   |
| 0.0039        | 13.1  | 4100 | 0.0416          | 0.9326    | 0.9467 | 0.9396 | 0.9944   |
| 0.0039        | 13.42 | 4200 | 0.0418          | 0.9327    | 0.9488 | 0.9407 | 0.9948   |
| 0.0039        | 13.74 | 4300 | 0.0423          | 0.9345    | 0.9460 | 0.9402 | 0.9946   |
| 0.0039        | 14.06 | 4400 | 0.0419          | 0.9286    | 0.9467 | 0.9376 | 0.9947   |
| 0.0022        | 14.38 | 4500 | 0.0426          | 0.9371    | 0.9438 | 0.9404 | 0.9945   |
| 0.0022        | 14.7  | 4600 | 0.0424          | 0.9371    | 0.9445 | 0.9408 | 0.9947   |
| 0.0022        | 15.02 | 4700 | 0.0427          | 0.9372    | 0.9467 | 0.9419 | 0.9947   |
| 0.0022        | 15.34 | 4800 | 0.0431          | 0.9339    | 0.9460 | 0.9399 | 0.9945   |
| 0.0022        | 15.65 | 4900 | 0.0431          | 0.9346    | 0.9467 | 0.9406 | 0.9946   |
| 0.0015        | 15.97 | 5000 | 0.0434          | 0.9324    | 0.9445 | 0.9384 | 0.9945   |


### Framework versions

- Transformers 4.20.0.dev0
- Pytorch 1.11.0+cu113
- Datasets 2.2.2
- Tokenizers 0.12.1
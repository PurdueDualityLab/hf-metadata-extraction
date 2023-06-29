---
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- accuracy
- wer
model-index:
- name: model_broadclass_onSet1.1
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# model_broadclass_onSet1.1

This model is a fine-tuned version of [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.2469
- 0 Precision: 1.0
- 0 Recall: 1.0
- 0 F1-score: 1.0
- 0 Support: 24
- 1 Precision: 1.0
- 1 Recall: 1.0
- 1 F1-score: 1.0
- 1 Support: 39
- 2 Precision: 1.0
- 2 Recall: 1.0
- 2 F1-score: 1.0
- 2 Support: 23
- 3 Precision: 1.0
- 3 Recall: 1.0
- 3 F1-score: 1.0
- 3 Support: 12
- Accuracy: 1.0
- Macro avg Precision: 1.0
- Macro avg Recall: 1.0
- Macro avg F1-score: 1.0
- Macro avg Support: 98
- Weighted avg Precision: 1.0
- Weighted avg Recall: 1.0
- Weighted avg F1-score: 1.0
- Weighted avg Support: 98
- Wer: 0.2423
- Mtrix: [[0, 1, 2, 3], [0, 24, 0, 0, 0], [1, 0, 39, 0, 0], [2, 0, 0, 23, 0], [3, 0, 0, 0, 12]]

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0003
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 16
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 200
- num_epochs: 80
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | 0 Precision | 0 Recall | 0 F1-score | 0 Support | 1 Precision | 1 Recall | 1 F1-score | 1 Support | 2 Precision | 2 Recall | 2 F1-score | 2 Support | 3 Precision | 3 Recall | 3 F1-score | 3 Support | Accuracy | Macro avg Precision | Macro avg Recall | Macro avg F1-score | Macro avg Support | Weighted avg Precision | Weighted avg Recall | Weighted avg F1-score | Weighted avg Support | Wer    | Mtrix                                                                                   |
|:-------------:|:-----:|:----:|:---------------:|:-----------:|:--------:|:----------:|:---------:|:-----------:|:--------:|:----------:|:---------:|:-----------:|:--------:|:----------:|:---------:|:-----------:|:--------:|:----------:|:---------:|:--------:|:-------------------:|:----------------:|:------------------:|:-----------------:|:----------------------:|:-------------------:|:---------------------:|:--------------------:|:------:|:---------------------------------------------------------------------------------------:|
| 2.3722        | 4.16  | 100  | 2.1950          | 0.2449      | 1.0      | 0.3934     | 24        | 0.0         | 0.0      | 0.0        | 39        | 0.0         | 0.0      | 0.0        | 23        | 0.0         | 0.0      | 0.0        | 12        | 0.2449   | 0.0612              | 0.25             | 0.0984             | 98                | 0.0600                 | 0.2449              | 0.0964                | 98                   | 0.9879 | [[0, 1, 2, 3], [0, 24, 0, 0, 0], [1, 39, 0, 0, 0], [2, 23, 0, 0, 0], [3, 12, 0, 0, 0]]  |
| 2.2944        | 8.33  | 200  | 2.1537          | 0.2449      | 1.0      | 0.3934     | 24        | 0.0         | 0.0      | 0.0        | 39        | 0.0         | 0.0      | 0.0        | 23        | 0.0         | 0.0      | 0.0        | 12        | 0.2449   | 0.0612              | 0.25             | 0.0984             | 98                | 0.0600                 | 0.2449              | 0.0964                | 98                   | 0.9879 | [[0, 1, 2, 3], [0, 24, 0, 0, 0], [1, 39, 0, 0, 0], [2, 23, 0, 0, 0], [3, 12, 0, 0, 0]]  |
| 1.9927        | 12.49 | 300  | 1.8879          | 0.2449      | 1.0      | 0.3934     | 24        | 0.0         | 0.0      | 0.0        | 39        | 0.0         | 0.0      | 0.0        | 23        | 0.0         | 0.0      | 0.0        | 12        | 0.2449   | 0.0612              | 0.25             | 0.0984             | 98                | 0.0600                 | 0.2449              | 0.0964                | 98                   | 0.9879 | [[0, 1, 2, 3], [0, 24, 0, 0, 0], [1, 39, 0, 0, 0], [2, 23, 0, 0, 0], [3, 12, 0, 0, 0]]  |
| 1.7175        | 16.65 | 400  | 1.6374          | 0.2449      | 1.0      | 0.3934     | 24        | 0.0         | 0.0      | 0.0        | 39        | 0.0         | 0.0      | 0.0        | 23        | 0.0         | 0.0      | 0.0        | 12        | 0.2449   | 0.0612              | 0.25             | 0.0984             | 98                | 0.0600                 | 0.2449              | 0.0964                | 98                   | 0.9879 | [[0, 1, 2, 3], [0, 24, 0, 0, 0], [1, 39, 0, 0, 0], [2, 23, 0, 0, 0], [3, 12, 0, 0, 0]]  |
| 1.6065        | 20.82 | 500  | 1.5619          | 0.2449      | 1.0      | 0.3934     | 24        | 0.0         | 0.0      | 0.0        | 39        | 0.0         | 0.0      | 0.0        | 23        | 0.0         | 0.0      | 0.0        | 12        | 0.2449   | 0.0612              | 0.25             | 0.0984             | 98                | 0.0600                 | 0.2449              | 0.0964                | 98                   | 0.9879 | [[0, 1, 2, 3], [0, 24, 0, 0, 0], [1, 39, 0, 0, 0], [2, 23, 0, 0, 0], [3, 12, 0, 0, 0]]  |
| 1.5362        | 24.98 | 600  | 1.5019          | 0.2449      | 1.0      | 0.3934     | 24        | 0.0         | 0.0      | 0.0        | 39        | 0.0         | 0.0      | 0.0        | 23        | 0.0         | 0.0      | 0.0        | 12        | 0.2449   | 0.0612              | 0.25             | 0.0984             | 98                | 0.0600                 | 0.2449              | 0.0964                | 98                   | 0.9879 | [[0, 1, 2, 3], [0, 24, 0, 0, 0], [1, 39, 0, 0, 0], [2, 23, 0, 0, 0], [3, 12, 0, 0, 0]]  |
| 1.5599        | 29.16 | 700  | 1.4858          | 0.2449      | 1.0      | 0.3934     | 24        | 0.0         | 0.0      | 0.0        | 39        | 0.0         | 0.0      | 0.0        | 23        | 0.0         | 0.0      | 0.0        | 12        | 0.2449   | 0.0612              | 0.25             | 0.0984             | 98                | 0.0600                 | 0.2449              | 0.0964                | 98                   | 0.9879 | [[0, 1, 2, 3], [0, 24, 0, 0, 0], [1, 39, 0, 0, 0], [2, 23, 0, 0, 0], [3, 12, 0, 0, 0]]  |
| 1.5344        | 33.33 | 800  | 1.4721          | 0.2759      | 1.0      | 0.4324     | 24        | 1.0         | 0.2821   | 0.4400     | 39        | 0.0         | 0.0      | 0.0        | 23        | 0.0         | 0.0      | 0.0        | 12        | 0.3571   | 0.3190              | 0.3205           | 0.2181             | 98                | 0.4655                 | 0.3571              | 0.2810                | 98                   | 0.9919 | [[0, 1, 2, 3], [0, 24, 0, 0, 0], [1, 28, 11, 0, 0], [2, 23, 0, 0, 0], [3, 12, 0, 0, 0]] |
| 1.4024        | 37.49 | 900  | 1.3532          | 1.0         | 1.0      | 1.0        | 24        | 1.0         | 1.0      | 1.0        | 39        | 1.0         | 1.0      | 1.0        | 23        | 1.0         | 1.0      | 1.0        | 12        | 1.0      | 1.0                 | 1.0              | 1.0                | 98                | 1.0                    | 1.0                 | 1.0                   | 98                   | 0.9742 | [[0, 1, 2, 3], [0, 24, 0, 0, 0], [1, 0, 39, 0, 0], [2, 0, 0, 23, 0], [3, 0, 0, 0, 12]]  |
| 0.9429        | 41.65 | 1000 | 0.9455          | 0.96        | 1.0      | 0.9796     | 24        | 0.9744      | 0.9744   | 0.9744     | 39        | 1.0         | 0.9565   | 0.9778     | 23        | 1.0         | 1.0      | 1.0        | 12        | 0.9796   | 0.9836              | 0.9827           | 0.9829             | 98                | 0.9800                 | 0.9796              | 0.9796                | 98                   | 0.9084 | [[0, 1, 2, 3], [0, 24, 0, 0, 0], [1, 1, 38, 0, 0], [2, 0, 1, 22, 0], [3, 0, 0, 0, 12]]  |
| 0.8955        | 45.82 | 1100 | 0.8890          | 0.96        | 1.0      | 0.9796     | 24        | 1.0         | 0.9744   | 0.9870     | 39        | 1.0         | 1.0      | 1.0        | 23        | 1.0         | 1.0      | 1.0        | 12        | 0.9898   | 0.99                | 0.9936           | 0.9917             | 98                | 0.9902                 | 0.9898              | 0.9898                | 98                   | 0.9246 | [[0, 1, 2, 3], [0, 24, 0, 0, 0], [1, 1, 38, 0, 0], [2, 0, 0, 23, 0], [3, 0, 0, 0, 12]]  |
| 0.8708        | 49.98 | 1200 | 0.8304          | 1.0         | 1.0      | 1.0        | 24        | 0.975       | 1.0      | 0.9873     | 39        | 1.0         | 0.9565   | 0.9778     | 23        | 1.0         | 1.0      | 1.0        | 12        | 0.9898   | 0.9938              | 0.9891           | 0.9913             | 98                | 0.9901                 | 0.9898              | 0.9897                | 98                   | 0.9272 | [[0, 1, 2, 3], [0, 24, 0, 0, 0], [1, 0, 39, 0, 0], [2, 0, 1, 22, 0], [3, 0, 0, 0, 12]]  |
| 0.8671        | 54.16 | 1300 | 0.8028          | 0.96        | 1.0      | 0.9796     | 24        | 1.0         | 1.0      | 1.0        | 39        | 1.0         | 0.9565   | 0.9778     | 23        | 1.0         | 1.0      | 1.0        | 12        | 0.9898   | 0.99                | 0.9891           | 0.9893             | 98                | 0.9902                 | 0.9898              | 0.9898                | 98                   | 0.9211 | [[0, 1, 2, 3], [0, 24, 0, 0, 0], [1, 0, 39, 0, 0], [2, 1, 0, 22, 0], [3, 0, 0, 0, 12]]  |
| 0.8383        | 58.33 | 1400 | 0.7804          | 1.0         | 1.0      | 1.0        | 24        | 1.0         | 1.0      | 1.0        | 39        | 1.0         | 1.0      | 1.0        | 23        | 1.0         | 1.0      | 1.0        | 12        | 1.0      | 1.0                 | 1.0              | 1.0                | 98                | 1.0                    | 1.0                 | 1.0                   | 98                   | 0.9170 | [[0, 1, 2, 3], [0, 24, 0, 0, 0], [1, 0, 39, 0, 0], [2, 0, 0, 23, 0], [3, 0, 0, 0, 12]]  |
| 0.7872        | 62.49 | 1500 | 0.7745          | 0.96        | 1.0      | 0.9796     | 24        | 1.0         | 0.9744   | 0.9870     | 39        | 1.0         | 1.0      | 1.0        | 23        | 1.0         | 1.0      | 1.0        | 12        | 0.9898   | 0.99                | 0.9936           | 0.9917             | 98                | 0.9902                 | 0.9898              | 0.9898                | 98                   | 0.9439 | [[0, 1, 2, 3], [0, 24, 0, 0, 0], [1, 1, 38, 0, 0], [2, 0, 0, 23, 0], [3, 0, 0, 0, 12]]  |
| 0.7538        | 66.65 | 1600 | 0.7141          | 0.96        | 1.0      | 0.9796     | 24        | 1.0         | 0.9744   | 0.9870     | 39        | 1.0         | 1.0      | 1.0        | 23        | 1.0         | 1.0      | 1.0        | 12        | 0.9898   | 0.99                | 0.9936           | 0.9917             | 98                | 0.9902                 | 0.9898              | 0.9898                | 98                   | 0.9267 | [[0, 1, 2, 3], [0, 24, 0, 0, 0], [1, 1, 38, 0, 0], [2, 0, 0, 23, 0], [3, 0, 0, 0, 12]]  |
| 0.6439        | 70.82 | 1700 | 0.5818          | 1.0         | 1.0      | 1.0        | 24        | 1.0         | 1.0      | 1.0        | 39        | 1.0         | 1.0      | 1.0        | 23        | 1.0         | 1.0      | 1.0        | 12        | 1.0      | 1.0                 | 1.0              | 1.0                | 98                | 1.0                    | 1.0                 | 1.0                   | 98                   | 0.8574 | [[0, 1, 2, 3], [0, 24, 0, 0, 0], [1, 0, 39, 0, 0], [2, 0, 0, 23, 0], [3, 0, 0, 0, 12]]  |
| 0.5295        | 74.98 | 1800 | 0.3775          | 1.0         | 1.0      | 1.0        | 24        | 1.0         | 1.0      | 1.0        | 39        | 1.0         | 1.0      | 1.0        | 23        | 1.0         | 1.0      | 1.0        | 12        | 1.0      | 1.0                 | 1.0              | 1.0                | 98                | 1.0                    | 1.0                 | 1.0                   | 98                   | 0.4633 | [[0, 1, 2, 3], [0, 24, 0, 0, 0], [1, 0, 39, 0, 0], [2, 0, 0, 23, 0], [3, 0, 0, 0, 12]]  |
| 0.4184        | 79.16 | 1900 | 0.2507          | 1.0         | 1.0      | 1.0        | 24        | 1.0         | 1.0      | 1.0        | 39        | 1.0         | 1.0      | 1.0        | 23        | 1.0         | 1.0      | 1.0        | 12        | 1.0      | 1.0                 | 1.0              | 1.0                | 98                | 1.0                    | 1.0                 | 1.0                   | 98                   | 0.2529 | [[0, 1, 2, 3], [0, 24, 0, 0, 0], [1, 0, 39, 0, 0], [2, 0, 0, 23, 0], [3, 0, 0, 0, 12]]  |


### Framework versions

- Transformers 4.25.1
- Pytorch 1.13.0+cu116
- Datasets 2.8.0
- Tokenizers 0.13.2
---
license: apache-2.0
tags:
- generated_from_trainer
metrics:
- accuracy
- wer
model-index:
- name: model_broadclass_onSet4
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# model_broadclass_onSet4

This model is a fine-tuned version of [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1340
- 0 Precision: 1.0
- 0 Recall: 0.9615
- 0 F1-score: 0.9804
- 0 Support: 26
- 1 Precision: 1.0
- 1 Recall: 1.0
- 1 F1-score: 1.0
- 1 Support: 32
- 2 Precision: 1.0
- 2 Recall: 0.9643
- 2 F1-score: 0.9818
- 2 Support: 28
- 3 Precision: 0.8462
- 3 Recall: 1.0
- 3 F1-score: 0.9167
- 3 Support: 11
- Accuracy: 0.9794
- Macro avg Precision: 0.9615
- Macro avg Recall: 0.9815
- Macro avg F1-score: 0.9697
- Macro avg Support: 97
- Weighted avg Precision: 0.9826
- Weighted avg Recall: 0.9794
- Weighted avg F1-score: 0.9800
- Weighted avg Support: 97
- Wer: 0.1098
- Mtrix: [[0, 1, 2, 3], [0, 25, 0, 0, 1], [1, 0, 32, 0, 0], [2, 0, 0, 27, 1], [3, 0, 0, 0, 11]]

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

| Training Loss | Epoch | Step | Validation Loss | 0 Precision | 0 Recall | 0 F1-score | 0 Support | 1 Precision | 1 Recall | 1 F1-score | 1 Support | 2 Precision | 2 Recall | 2 F1-score | 2 Support | 3 Precision | 3 Recall | 3 F1-score | 3 Support | Accuracy | Macro avg Precision | Macro avg Recall | Macro avg F1-score | Macro avg Support | Weighted avg Precision | Weighted avg Recall | Weighted avg F1-score | Weighted avg Support | Wer    | Mtrix                                                                                  |
|:-------------:|:-----:|:----:|:---------------:|:-----------:|:--------:|:----------:|:---------:|:-----------:|:--------:|:----------:|:---------:|:-----------:|:--------:|:----------:|:---------:|:-----------:|:--------:|:----------:|:---------:|:--------:|:-------------------:|:----------------:|:------------------:|:-----------------:|:----------------------:|:-------------------:|:---------------------:|:--------------------:|:------:|:--------------------------------------------------------------------------------------:|
| 2.337         | 4.16  | 100  | 2.1761          | 0.2680      | 1.0      | 0.4228     | 26        | 0.0         | 0.0      | 0.0        | 32        | 0.0         | 0.0      | 0.0        | 28        | 0.0         | 0.0      | 0.0        | 11        | 0.2680   | 0.0670              | 0.25             | 0.1057             | 97                | 0.0718                 | 0.2680              | 0.1133                | 97                   | 0.9869 | [[0, 1, 2, 3], [0, 26, 0, 0, 0], [1, 32, 0, 0, 0], [2, 28, 0, 0, 0], [3, 11, 0, 0, 0]] |
| 2.2604        | 8.33  | 200  | 2.0783          | 0.2680      | 1.0      | 0.4228     | 26        | 0.0         | 0.0      | 0.0        | 32        | 0.0         | 0.0      | 0.0        | 28        | 0.0         | 0.0      | 0.0        | 11        | 0.2680   | 0.0670              | 0.25             | 0.1057             | 97                | 0.0718                 | 0.2680              | 0.1133                | 97                   | 0.9869 | [[0, 1, 2, 3], [0, 26, 0, 0, 0], [1, 32, 0, 0, 0], [2, 28, 0, 0, 0], [3, 11, 0, 0, 0]] |
| 1.9239        | 12.49 | 300  | 1.8395          | 0.2680      | 1.0      | 0.4228     | 26        | 0.0         | 0.0      | 0.0        | 32        | 0.0         | 0.0      | 0.0        | 28        | 0.0         | 0.0      | 0.0        | 11        | 0.2680   | 0.0670              | 0.25             | 0.1057             | 97                | 0.0718                 | 0.2680              | 0.1133                | 97                   | 0.9869 | [[0, 1, 2, 3], [0, 26, 0, 0, 0], [1, 32, 0, 0, 0], [2, 28, 0, 0, 0], [3, 11, 0, 0, 0]] |
| 1.7002        | 16.65 | 400  | 1.7194          | 0.2680      | 1.0      | 0.4228     | 26        | 0.0         | 0.0      | 0.0        | 32        | 0.0         | 0.0      | 0.0        | 28        | 0.0         | 0.0      | 0.0        | 11        | 0.2680   | 0.0670              | 0.25             | 0.1057             | 97                | 0.0718                 | 0.2680              | 0.1133                | 97                   | 0.9869 | [[0, 1, 2, 3], [0, 26, 0, 0, 0], [1, 32, 0, 0, 0], [2, 28, 0, 0, 0], [3, 11, 0, 0, 0]] |
| 1.611         | 20.82 | 500  | 1.5619          | 0.2680      | 1.0      | 0.4228     | 26        | 0.0         | 0.0      | 0.0        | 32        | 0.0         | 0.0      | 0.0        | 28        | 0.0         | 0.0      | 0.0        | 11        | 0.2680   | 0.0670              | 0.25             | 0.1057             | 97                | 0.0718                 | 0.2680              | 0.1133                | 97                   | 0.9869 | [[0, 1, 2, 3], [0, 26, 0, 0, 0], [1, 32, 0, 0, 0], [2, 28, 0, 0, 0], [3, 11, 0, 0, 0]] |
| 1.486         | 24.98 | 600  | 1.5283          | 0.2680      | 1.0      | 0.4228     | 26        | 0.0         | 0.0      | 0.0        | 32        | 0.0         | 0.0      | 0.0        | 28        | 0.0         | 0.0      | 0.0        | 11        | 0.2680   | 0.0670              | 0.25             | 0.1057             | 97                | 0.0718                 | 0.2680              | 0.1133                | 97                   | 0.9869 | [[0, 1, 2, 3], [0, 26, 0, 0, 0], [1, 32, 0, 0, 0], [2, 28, 0, 0, 0], [3, 11, 0, 0, 0]] |
| 1.6085        | 29.16 | 700  | 1.5041          | 0.2680      | 1.0      | 0.4228     | 26        | 0.0         | 0.0      | 0.0        | 32        | 0.0         | 0.0      | 0.0        | 28        | 0.0         | 0.0      | 0.0        | 11        | 0.2680   | 0.0670              | 0.25             | 0.1057             | 97                | 0.0718                 | 0.2680              | 0.1133                | 97                   | 0.9869 | [[0, 1, 2, 3], [0, 26, 0, 0, 0], [1, 32, 0, 0, 0], [2, 28, 0, 0, 0], [3, 11, 0, 0, 0]] |
| 1.5607        | 33.33 | 800  | 1.4456          | 0.2680      | 1.0      | 0.4228     | 26        | 0.0         | 0.0      | 0.0        | 32        | 0.0         | 0.0      | 0.0        | 28        | 0.0         | 0.0      | 0.0        | 11        | 0.2680   | 0.0670              | 0.25             | 0.1057             | 97                | 0.0718                 | 0.2680              | 0.1133                | 97                   | 0.9945 | [[0, 1, 2, 3], [0, 26, 0, 0, 0], [1, 32, 0, 0, 0], [2, 28, 0, 0, 0], [3, 11, 0, 0, 0]] |
| 1.3499        | 37.49 | 900  | 1.2898          | 0.2680      | 1.0      | 0.4228     | 26        | 0.0         | 0.0      | 0.0        | 32        | 0.0         | 0.0      | 0.0        | 28        | 0.0         | 0.0      | 0.0        | 11        | 0.2680   | 0.0670              | 0.25             | 0.1057             | 97                | 0.0718                 | 0.2680              | 0.1133                | 97                   | 0.9970 | [[0, 1, 2, 3], [0, 26, 0, 0, 0], [1, 32, 0, 0, 0], [2, 28, 0, 0, 0], [3, 11, 0, 0, 0]] |
| 0.9722        | 41.65 | 1000 | 0.9757          | 0.3133      | 1.0      | 0.4771     | 26        | 1.0         | 0.1562   | 0.2703     | 32        | 1.0         | 0.1786   | 0.3030     | 28        | 1.0         | 0.3636   | 0.5333     | 11        | 0.4124   | 0.8283              | 0.4246           | 0.3959             | 97                | 0.8159                 | 0.4124              | 0.3650                | 97                   | 0.9612 | [[0, 1, 2, 3], [0, 26, 0, 0, 0], [1, 27, 5, 0, 0], [2, 23, 0, 5, 0], [3, 7, 0, 0, 4]]  |
| 0.9679        | 45.82 | 1100 | 0.9452          | 0.4333      | 1.0      | 0.6047     | 26        | 0.9630      | 0.8125   | 0.8814     | 32        | 1.0         | 0.3214   | 0.4865     | 28        | 1.0         | 0.0909   | 0.1667     | 11        | 0.6392   | 0.8491              | 0.5562           | 0.5348             | 97                | 0.8359                 | 0.6392              | 0.6122                | 97                   | 0.9406 | [[0, 1, 2, 3], [0, 26, 0, 0, 0], [1, 6, 26, 0, 0], [2, 18, 1, 9, 0], [3, 10, 0, 0, 1]] |
| 0.9206        | 49.98 | 1200 | 0.9031          | 0.5909      | 1.0      | 0.7429     | 26        | 1.0         | 0.9062   | 0.9508     | 32        | 1.0         | 0.7143   | 0.8333     | 28        | 1.0         | 0.3636   | 0.5333     | 11        | 0.8144   | 0.8977              | 0.7460           | 0.7651             | 97                | 0.8903                 | 0.8144              | 0.8138                | 97                   | 0.9250 | [[0, 1, 2, 3], [0, 26, 0, 0, 0], [1, 3, 29, 0, 0], [2, 8, 0, 20, 0], [3, 7, 0, 0, 4]]  |
| 0.9223        | 54.16 | 1300 | 0.8607          | 0.8125      | 1.0      | 0.8966     | 26        | 1.0         | 0.875    | 0.9333     | 32        | 1.0         | 0.9643   | 0.9818     | 28        | 1.0         | 0.9091   | 0.9524     | 11        | 0.9381   | 0.9531              | 0.9371           | 0.9410             | 97                | 0.9497                 | 0.9381              | 0.9396                | 97                   | 0.9366 | [[0, 1, 2, 3], [0, 26, 0, 0, 0], [1, 4, 28, 0, 0], [2, 1, 0, 27, 0], [3, 1, 0, 0, 10]] |
| 0.8407        | 58.33 | 1400 | 0.8011          | 0.8929      | 0.9615   | 0.9259     | 26        | 1.0         | 0.9688   | 0.9841     | 32        | 1.0         | 0.8929   | 0.9434     | 28        | 0.8462      | 1.0      | 0.9167     | 11        | 0.9485   | 0.9348              | 0.9558           | 0.9425             | 97                | 0.9538                 | 0.9485              | 0.9491                | 97                   | 0.9381 | [[0, 1, 2, 3], [0, 25, 0, 0, 1], [1, 1, 31, 0, 0], [2, 2, 0, 25, 1], [3, 0, 0, 0, 11]] |
| 0.7359        | 62.49 | 1500 | 0.7210          | 0.8966      | 1.0      | 0.9455     | 26        | 1.0         | 0.9375   | 0.9677     | 32        | 1.0         | 0.9286   | 0.9630     | 28        | 0.9167      | 1.0      | 0.9565     | 11        | 0.9588   | 0.9533              | 0.9665           | 0.9582             | 97                | 0.9628                 | 0.9588              | 0.9591                | 97                   | 0.9220 | [[0, 1, 2, 3], [0, 26, 0, 0, 0], [1, 2, 30, 0, 0], [2, 1, 0, 26, 1], [3, 0, 0, 0, 11]] |
| 0.5479        | 66.65 | 1600 | 0.4813          | 1.0         | 1.0      | 1.0        | 26        | 1.0         | 1.0      | 1.0        | 32        | 1.0         | 0.9643   | 0.9818     | 28        | 0.9167      | 1.0      | 0.9565     | 11        | 0.9897   | 0.9792              | 0.9911           | 0.9846             | 97                | 0.9905                 | 0.9897              | 0.9898                | 97                   | 0.7447 | [[0, 1, 2, 3], [0, 26, 0, 0, 0], [1, 0, 32, 0, 0], [2, 0, 0, 27, 1], [3, 0, 0, 0, 11]] |
| 0.2617        | 70.82 | 1700 | 0.2138          | 1.0         | 1.0      | 1.0        | 26        | 1.0         | 1.0      | 1.0        | 32        | 1.0         | 0.9643   | 0.9818     | 28        | 0.9167      | 1.0      | 0.9565     | 11        | 0.9897   | 0.9792              | 0.9911           | 0.9846             | 97                | 0.9905                 | 0.9897              | 0.9898                | 97                   | 0.1692 | [[0, 1, 2, 3], [0, 26, 0, 0, 0], [1, 0, 32, 0, 0], [2, 0, 0, 27, 1], [3, 0, 0, 0, 11]] |
| 0.2186        | 74.98 | 1800 | 0.1412          | 1.0         | 1.0      | 1.0        | 26        | 1.0         | 1.0      | 1.0        | 32        | 1.0         | 0.9643   | 0.9818     | 28        | 0.9167      | 1.0      | 0.9565     | 11        | 0.9897   | 0.9792              | 0.9911           | 0.9846             | 97                | 0.9905                 | 0.9897              | 0.9898                | 97                   | 0.1269 | [[0, 1, 2, 3], [0, 26, 0, 0, 0], [1, 0, 32, 0, 0], [2, 0, 0, 27, 1], [3, 0, 0, 0, 11]] |
| 0.2303        | 79.16 | 1900 | 0.1344          | 1.0         | 0.9615   | 0.9804     | 26        | 1.0         | 1.0      | 1.0        | 32        | 1.0         | 0.9643   | 0.9818     | 28        | 0.8462      | 1.0      | 0.9167     | 11        | 0.9794   | 0.9615              | 0.9815           | 0.9697             | 97                | 0.9826                 | 0.9794              | 0.9800                | 97                   | 0.1113 | [[0, 1, 2, 3], [0, 25, 0, 0, 1], [1, 0, 32, 0, 0], [2, 0, 0, 27, 1], [3, 0, 0, 0, 11]] |


### Framework versions

- Transformers 4.25.1
- Pytorch 1.13.0+cu116
- Datasets 2.8.0
- Tokenizers 0.13.2
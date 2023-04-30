---
license: apache-2.0
tags:
- generated_from_trainer
model-index:
- name: wav2vec2-base-timit-demo-colab
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# wav2vec2-base-timit-demo-colab

This model is a fine-tuned version of [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.6259
- Wer: 0.3544

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 4
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 1000
- num_epochs: 30
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Wer    |
|:-------------:|:-----:|:-----:|:---------------:|:------:|
| 3.6744        | 0.5   | 500   | 2.9473          | 1.0    |
| 1.4535        | 1.01  | 1000  | 0.7774          | 0.6254 |
| 0.7376        | 1.51  | 1500  | 0.6923          | 0.5712 |
| 0.5848        | 2.01  | 2000  | 0.5445          | 0.5023 |
| 0.4492        | 2.51  | 2500  | 0.5148          | 0.4958 |
| 0.4006        | 3.02  | 3000  | 0.5283          | 0.4781 |
| 0.3319        | 3.52  | 3500  | 0.5196          | 0.4628 |
| 0.3424        | 4.02  | 4000  | 0.5285          | 0.4551 |
| 0.2772        | 4.52  | 4500  | 0.5060          | 0.4532 |
| 0.2724        | 5.03  | 5000  | 0.5216          | 0.4422 |
| 0.2375        | 5.53  | 5500  | 0.5376          | 0.4443 |
| 0.2279        | 6.03  | 6000  | 0.6051          | 0.4308 |
| 0.2091        | 6.53  | 6500  | 0.5084          | 0.4423 |
| 0.2029        | 7.04  | 7000  | 0.5083          | 0.4242 |
| 0.1784        | 7.54  | 7500  | 0.6123          | 0.4297 |
| 0.1774        | 8.04  | 8000  | 0.5749          | 0.4339 |
| 0.1542        | 8.54  | 8500  | 0.5110          | 0.4033 |
| 0.1638        | 9.05  | 9000  | 0.6324          | 0.4318 |
| 0.1493        | 9.55  | 9500  | 0.6100          | 0.4152 |
| 0.1591        | 10.05 | 10000 | 0.5508          | 0.4022 |
| 0.1304        | 10.55 | 10500 | 0.5090          | 0.4054 |
| 0.1234        | 11.06 | 11000 | 0.6282          | 0.4093 |
| 0.1218        | 11.56 | 11500 | 0.5817          | 0.3941 |
| 0.121         | 12.06 | 12000 | 0.5741          | 0.3999 |
| 0.1073        | 12.56 | 12500 | 0.5818          | 0.4149 |
| 0.104         | 13.07 | 13000 | 0.6492          | 0.3953 |
| 0.0934        | 13.57 | 13500 | 0.5393          | 0.4083 |
| 0.0961        | 14.07 | 14000 | 0.5510          | 0.3919 |
| 0.0965        | 14.57 | 14500 | 0.5896          | 0.3992 |
| 0.0921        | 15.08 | 15000 | 0.5554          | 0.3947 |
| 0.0751        | 15.58 | 15500 | 0.6312          | 0.3934 |
| 0.0805        | 16.08 | 16000 | 0.6732          | 0.3948 |
| 0.0742        | 16.58 | 16500 | 0.5990          | 0.3884 |
| 0.0708        | 17.09 | 17000 | 0.6186          | 0.3869 |
| 0.0679        | 17.59 | 17500 | 0.5837          | 0.3848 |
| 0.072         | 18.09 | 18000 | 0.5831          | 0.3775 |
| 0.0597        | 18.59 | 18500 | 0.6562          | 0.3843 |
| 0.0612        | 19.1  | 19000 | 0.6298          | 0.3756 |
| 0.0514        | 19.6  | 19500 | 0.6746          | 0.3720 |
| 0.061         | 20.1  | 20000 | 0.6236          | 0.3788 |
| 0.054         | 20.6  | 20500 | 0.6012          | 0.3718 |
| 0.0521        | 21.11 | 21000 | 0.6053          | 0.3778 |
| 0.0494        | 21.61 | 21500 | 0.6154          | 0.3772 |
| 0.0468        | 22.11 | 22000 | 0.6052          | 0.3747 |
| 0.0413        | 22.61 | 22500 | 0.5877          | 0.3716 |
| 0.0424        | 23.12 | 23000 | 0.5786          | 0.3658 |
| 0.0403        | 23.62 | 23500 | 0.5828          | 0.3658 |
| 0.0391        | 24.12 | 24000 | 0.5913          | 0.3685 |
| 0.0312        | 24.62 | 24500 | 0.5850          | 0.3625 |
| 0.0316        | 25.13 | 25000 | 0.6029          | 0.3611 |
| 0.0282        | 25.63 | 25500 | 0.6312          | 0.3624 |
| 0.0328        | 26.13 | 26000 | 0.6312          | 0.3621 |
| 0.0258        | 26.63 | 26500 | 0.5891          | 0.3581 |
| 0.0256        | 27.14 | 27000 | 0.6259          | 0.3546 |
| 0.0255        | 27.64 | 27500 | 0.6315          | 0.3587 |
| 0.0249        | 28.14 | 28000 | 0.6547          | 0.3579 |
| 0.025         | 28.64 | 28500 | 0.6237          | 0.3565 |
| 0.0228        | 29.15 | 29000 | 0.6187          | 0.3559 |
| 0.0209        | 29.65 | 29500 | 0.6259          | 0.3544 |


### Framework versions

- Transformers 4.11.3
- Pytorch 1.10.0+cu102
- Datasets 1.18.3
- Tokenizers 0.10.3
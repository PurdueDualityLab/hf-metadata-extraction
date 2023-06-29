---
license: apache-2.0
tags:
- generated_from_trainer
model-index:
- name: wav2vec2-large-xlsr-korean-demo-test2
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# wav2vec2-large-xlsr-korean-demo-test2

This model is a fine-tuned version of [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 1.0566
- Wer: 0.5224

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
- train_batch_size: 4
- eval_batch_size: 8
- seed: 42
- gradient_accumulation_steps: 2
- total_train_batch_size: 8
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_steps: 500
- num_epochs: 20
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Wer    |
|:-------------:|:-----:|:-----:|:---------------:|:------:|
| 31.2541       | 0.3   | 400   | 5.4002          | 1.0    |
| 4.9419        | 0.59  | 800   | 5.3336          | 1.0    |
| 4.8926        | 0.89  | 1200  | 5.0531          | 1.0    |
| 4.7218        | 1.19  | 1600  | 4.5172          | 1.0    |
| 4.0218        | 1.49  | 2000  | 3.1418          | 0.9518 |
| 3.0654        | 1.78  | 2400  | 2.4376          | 0.9041 |
| 2.6226        | 2.08  | 2800  | 2.0151          | 0.8643 |
| 2.2944        | 2.38  | 3200  | 1.8025          | 0.8290 |
| 2.1872        | 2.67  | 3600  | 1.6469          | 0.7962 |
| 2.0747        | 2.97  | 4000  | 1.5165          | 0.7714 |
| 1.8479        | 3.27  | 4400  | 1.4281          | 0.7694 |
| 1.8288        | 3.57  | 4800  | 1.3791          | 0.7326 |
| 1.801         | 3.86  | 5200  | 1.3328          | 0.7177 |
| 1.6723        | 4.16  | 5600  | 1.2954          | 0.7192 |
| 1.5925        | 4.46  | 6000  | 1.3137          | 0.6953 |
| 1.5709        | 4.75  | 6400  | 1.2086          | 0.6973 |
| 1.5294        | 5.05  | 6800  | 1.1811          | 0.6730 |
| 1.3844        | 5.35  | 7200  | 1.2053          | 0.6769 |
| 1.3906        | 5.65  | 7600  | 1.1287          | 0.6556 |
| 1.4088        | 5.94  | 8000  | 1.1251          | 0.6466 |
| 1.2989        | 6.24  | 8400  | 1.1577          | 0.6546 |
| 1.2523        | 6.54  | 8800  | 1.0643          | 0.6377 |
| 1.2651        | 6.84  | 9200  | 1.0865          | 0.6417 |
| 1.2209        | 7.13  | 9600  | 1.0981          | 0.6272 |
| 1.1435        | 7.43  | 10000 | 1.1195          | 0.6317 |
| 1.1616        | 7.73  | 10400 | 1.0672          | 0.6327 |
| 1.1272        | 8.02  | 10800 | 1.0413          | 0.6248 |
| 1.043         | 8.32  | 11200 | 1.0555          | 0.6233 |
| 1.0523        | 8.62  | 11600 | 1.0372          | 0.6178 |
| 1.0208        | 8.92  | 12000 | 1.0170          | 0.6128 |
| 0.9895        | 9.21  | 12400 | 1.0354          | 0.5934 |
| 0.95          | 9.51  | 12800 | 1.1019          | 0.6039 |
| 0.9705        | 9.81  | 13200 | 1.0229          | 0.5855 |
| 0.9202        | 10.1  | 13600 | 1.0364          | 0.5919 |
| 0.8644        | 10.4  | 14000 | 1.0721          | 0.5984 |
| 0.8641        | 10.7  | 14400 | 1.0383          | 0.5905 |
| 0.8924        | 11.0  | 14800 | 0.9947          | 0.5760 |
| 0.7914        | 11.29 | 15200 | 1.0270          | 0.5885 |
| 0.7882        | 11.59 | 15600 | 1.0271          | 0.5741 |
| 0.8116        | 11.89 | 16000 | 0.9937          | 0.5741 |
| 0.7584        | 12.18 | 16400 | 0.9924          | 0.5626 |
| 0.7051        | 12.48 | 16800 | 1.0023          | 0.5572 |
| 0.7232        | 12.78 | 17200 | 1.0479          | 0.5512 |
| 0.7149        | 13.08 | 17600 | 1.0475          | 0.5765 |
| 0.6579        | 13.37 | 18000 | 1.0218          | 0.5552 |
| 0.6615        | 13.67 | 18400 | 1.0339          | 0.5631 |
| 0.6629        | 13.97 | 18800 | 1.0239          | 0.5621 |
| 0.6221        | 14.26 | 19200 | 1.0331          | 0.5537 |
| 0.6159        | 14.56 | 19600 | 1.0640          | 0.5532 |
| 0.6032        | 14.86 | 20000 | 1.0192          | 0.5567 |
| 0.5748        | 15.16 | 20400 | 1.0093          | 0.5507 |
| 0.5614        | 15.45 | 20800 | 1.0458          | 0.5472 |
| 0.5626        | 15.75 | 21200 | 1.0318          | 0.5398 |
| 0.5429        | 16.05 | 21600 | 1.0112          | 0.5278 |
| 0.5407        | 16.34 | 22000 | 1.0120          | 0.5278 |
| 0.511         | 16.64 | 22400 | 1.0335          | 0.5249 |
| 0.5316        | 16.94 | 22800 | 1.0146          | 0.5348 |
| 0.4949        | 17.24 | 23200 | 1.0287          | 0.5388 |
| 0.496         | 17.53 | 23600 | 1.0229          | 0.5348 |
| 0.4986        | 17.83 | 24000 | 1.0094          | 0.5313 |
| 0.4787        | 18.13 | 24400 | 1.0620          | 0.5234 |
| 0.4508        | 18.42 | 24800 | 1.0401          | 0.5323 |
| 0.4754        | 18.72 | 25200 | 1.0543          | 0.5303 |
| 0.4584        | 19.02 | 25600 | 1.0433          | 0.5194 |
| 0.4431        | 19.32 | 26000 | 1.0597          | 0.5249 |
| 0.4448        | 19.61 | 26400 | 1.0548          | 0.5229 |
| 0.4475        | 19.91 | 26800 | 1.0566          | 0.5224 |


### Framework versions

- Transformers 4.21.1
- Pytorch 1.12.1+cu113
- Datasets 2.4.0
- Tokenizers 0.12.1
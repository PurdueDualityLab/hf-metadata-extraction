---
license: apache-2.0
tags:
- audio-classification
- generated_from_trainer
datasets:
- superb
metrics:
- accuracy
model-index:
- name: wav2vec2-base-ks-padpt3200
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# wav2vec2-base-ks-padpt3200

This model is a fine-tuned version of [facebook/wav2vec2-base](https://huggingface.co/facebook/wav2vec2-base) on the superb dataset.
It achieves the following results on the evaluation set:
- Loss: 1.2818
- Accuracy: 0.6200

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.003
- train_batch_size: 256
- eval_batch_size: 256
- seed: 0
- gradient_accumulation_steps: 4
- total_train_batch_size: 1024
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 10.0
- mixed_precision_training: Native AMP

### Training results

| Training Loss | Epoch | Step | Validation Loss | Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:--------:|
| 1.3802        | 1.0   | 50   | 1.5035          | 0.6121   |
| 1.0153        | 2.0   | 100  | 1.2818          | 0.6200   |
| 0.9105        | 3.0   | 150  | 1.3827          | 0.5380   |
| 0.8535        | 4.0   | 200  | 1.3513          | 0.5587   |
| 0.7982        | 5.0   | 250  | 1.4749          | 0.5068   |
| 0.7754        | 6.0   | 300  | 1.5109          | 0.5025   |
| 0.749         | 7.0   | 350  | 1.6198          | 0.4476   |
| 0.7497        | 8.0   | 400  | 1.5480          | 0.4850   |
| 0.7386        | 9.0   | 450  | 1.6052          | 0.4665   |
| 0.7185        | 10.0  | 500  | 1.6085          | 0.4734   |


### Framework versions

- Transformers 4.22.0.dev0
- Pytorch 1.11.0+cu115
- Datasets 2.4.0
- Tokenizers 0.12.1
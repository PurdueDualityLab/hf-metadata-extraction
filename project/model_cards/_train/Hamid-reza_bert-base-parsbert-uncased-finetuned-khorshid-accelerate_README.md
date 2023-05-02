---
tags:
- generated_from_trainer
model-index:
- name: bert-base-parsbert-uncased-finetuned-khorshid-accelerate
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bert-base-parsbert-uncased-finetuned-khorshid-accelerate

This model is a fine-tuned version of [HooshvareLab/bert-base-parsbert-uncased](https://huggingface.co/HooshvareLab/bert-base-parsbert-uncased) on [a sentiment analysis dataset of Hala-Khorshid T.V program](https://drive.google.com/file/d/1vztWibWDCeiueAqhl91hbJnQblskQkPq/view?usp=sharing) dataset.
We used 🤗 Accelerate to train the model.

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Training results

More information needed


### Framework versions

- Transformers 4.25.1
- Pytorch 1.13.1+cu116
- Datasets 2.8.0
- Tokenizers 0.13.2

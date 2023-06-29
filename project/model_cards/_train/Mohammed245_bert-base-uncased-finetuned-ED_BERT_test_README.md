---
license: apache-2.0
tags:
- generated_from_keras_callback
model-index:
- name: Mohammed245/bert-base-uncased-finetuned-ED_BERT_test
  results: []
---

<!-- This model card has been generated automatically according to the information Keras had access to. You should
probably proofread and complete it, then remove this comment. -->

# Mohammed245/bert-base-uncased-finetuned-ED_BERT_test

This model is a fine-tuned version of [bert-base-uncased](https://huggingface.co/bert-base-uncased) on an unknown dataset.
It achieves the following results on the evaluation set:
- Train Loss: 4.9663
- Validation Loss: 5.2474
- Epoch: 1

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- optimizer: {'name': 'AdamWeightDecay', 'learning_rate': 2e-05, 'decay': 0.0, 'beta_1': 0.9, 'beta_2': 0.999, 'epsilon': 1e-07, 'amsgrad': False, 'weight_decay_rate': 0.01}
- training_precision: float32

### Training results

| Train Loss | Validation Loss | Epoch |
|:----------:|:---------------:|:-----:|
| 6.2084     | 5.6239          | 0     |
| 4.9663     | 5.2474          | 1     |


### Framework versions

- Transformers 4.18.0
- TensorFlow 2.4.0
- Datasets 2.2.1
- Tokenizers 0.11.0
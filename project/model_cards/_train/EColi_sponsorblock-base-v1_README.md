---
tags:
- generated_from_trainer
model-index:
- name: out
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# out

This model is a fine-tuned version of [/1TB_SSD/SB_AI/out_epoch1/out/checkpoint-1115000/](https://huggingface.co//1TB_SSD/SB_AI/out_epoch1/out/checkpoint-1115000/) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0645

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 2518227880
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 2.0

### Training results

| Training Loss | Epoch | Step    | Validation Loss |
|:-------------:|:-----:|:-------:|:---------------:|
| 0.0867        | 0.07  | 75000   | 0.0742          |
| 0.0783        | 0.13  | 150000  | 0.0695          |
| 0.0719        | 0.2   | 225000  | 0.0732          |
| 0.0743        | 0.27  | 300000  | 0.0663          |
| 0.0659        | 0.34  | 375000  | 0.0686          |
| 0.0664        | 0.4   | 450000  | 0.0683          |
| 0.0637        | 0.47  | 525000  | 0.0680          |
| 0.0655        | 0.54  | 600000  | 0.0641          |
| 0.0676        | 0.6   | 675000  | 0.0644          |
| 0.0704        | 0.67  | 750000  | 0.0645          |
| 0.0687        | 0.74  | 825000  | 0.0610          |
| 0.059         | 0.81  | 900000  | 0.0652          |
| 0.0666        | 0.87  | 975000  | 0.0619          |
| 0.0624        | 0.94  | 1050000 | 0.0619          |
| 0.0625        | 1.01  | 1125000 | 0.0667          |
| 0.0614        | 1.03  | 1150000 | 0.0658          |
| 0.0597        | 1.05  | 1175000 | 0.0683          |
| 0.0629        | 1.07  | 1200000 | 0.0691          |
| 0.0603        | 1.1   | 1225000 | 0.0678          |
| 0.0601        | 1.12  | 1250000 | 0.0746          |
| 0.0606        | 1.14  | 1275000 | 0.0691          |
| 0.0671        | 1.16  | 1300000 | 0.0702          |
| 0.0625        | 1.19  | 1325000 | 0.0661          |
| 0.0617        | 1.21  | 1350000 | 0.0688          |
| 0.0579        | 1.23  | 1375000 | 0.0679          |
| 0.0663        | 1.25  | 1400000 | 0.0634          |
| 0.0583        | 1.28  | 1425000 | 0.0638          |
| 0.0623        | 1.3   | 1450000 | 0.0681          |
| 0.0615        | 1.32  | 1475000 | 0.0670          |
| 0.0592        | 1.34  | 1500000 | 0.0666          |
| 0.0626        | 1.37  | 1525000 | 0.0666          |
| 0.063         | 1.39  | 1550000 | 0.0647          |
| 0.0648        | 1.41  | 1575000 | 0.0653          |
| 0.0611        | 1.43  | 1600000 | 0.0700          |
| 0.0622        | 1.46  | 1625000 | 0.0634          |
| 0.0617        | 1.48  | 1650000 | 0.0651          |
| 0.0613        | 1.5   | 1675000 | 0.0634          |
| 0.0639        | 1.52  | 1700000 | 0.0661          |
| 0.0615        | 1.54  | 1725000 | 0.0644          |
| 0.0605        | 1.57  | 1750000 | 0.0662          |
| 0.0622        | 1.59  | 1775000 | 0.0656          |
| 0.0585        | 1.61  | 1800000 | 0.0633          |
| 0.0628        | 1.63  | 1825000 | 0.0625          |
| 0.0638        | 1.66  | 1850000 | 0.0662          |
| 0.0599        | 1.68  | 1875000 | 0.0664          |
| 0.0583        | 1.7   | 1900000 | 0.0668          |
| 0.0543        | 1.72  | 1925000 | 0.0631          |
| 0.06          | 1.75  | 1950000 | 0.0629          |
| 0.0615        | 1.77  | 1975000 | 0.0644          |
| 0.0587        | 1.79  | 2000000 | 0.0663          |
| 0.0647        | 1.81  | 2025000 | 0.0654          |
| 0.0604        | 1.84  | 2050000 | 0.0639          |
| 0.0641        | 1.86  | 2075000 | 0.0636          |
| 0.0604        | 1.88  | 2100000 | 0.0636          |
| 0.0654        | 1.9   | 2125000 | 0.0652          |
| 0.0588        | 1.93  | 2150000 | 0.0638          |
| 0.0616        | 1.95  | 2175000 | 0.0657          |
| 0.0598        | 1.97  | 2200000 | 0.0646          |
| 0.0633        | 1.99  | 2225000 | 0.0645          |


### Framework versions

- Transformers 4.15.0
- Pytorch 1.10.1+cu113
- Datasets 1.17.0
- Tokenizers 0.10.3
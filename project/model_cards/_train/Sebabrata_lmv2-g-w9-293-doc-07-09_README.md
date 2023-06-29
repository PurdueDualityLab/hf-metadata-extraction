---
license: cc-by-nc-sa-4.0
tags:
- generated_from_trainer
model-index:
- name: lmv2-g-w9-293-doc-07-09
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# lmv2-g-w9-293-doc-07-09

This model is a fine-tuned version of [microsoft/layoutlmv2-base-uncased](https://huggingface.co/microsoft/layoutlmv2-base-uncased) on the None dataset.
It achieves the following results on the evaluation set:
- Loss: 0.0031
- Address Precision: 1.0
- Address Recall: 1.0
- Address F1: 1.0
- Address Number: 59
- Business Name Precision: 0.9737
- Business Name Recall: 0.9737
- Business Name F1: 0.9737
- Business Name Number: 38
- City State Zip Code Precision: 1.0
- City State Zip Code Recall: 1.0
- City State Zip Code F1: 1.0
- City State Zip Code Number: 59
- Ein Precision: 0.9474
- Ein Recall: 0.9
- Ein F1: 0.9231
- Ein Number: 20
- List Account Number Precision: 1.0
- List Account Number Recall: 1.0
- List Account Number F1: 1.0
- List Account Number Number: 59
- Name Precision: 1.0
- Name Recall: 1.0
- Name F1: 1.0
- Name Number: 59
- Ssn Precision: 0.9268
- Ssn Recall: 0.9744
- Ssn F1: 0.9500
- Ssn Number: 39
- Overall Precision: 0.9850
- Overall Recall: 0.9880
- Overall F1: 0.9865
- Overall Accuracy: 0.9995

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 4e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: constant
- num_epochs: 30

### Training results

| Training Loss | Epoch | Step | Validation Loss | Address Precision | Address Recall | Address F1 | Address Number | Business Name Precision | Business Name Recall | Business Name F1 | Business Name Number | City State Zip Code Precision | City State Zip Code Recall | City State Zip Code F1 | City State Zip Code Number | Ein Precision | Ein Recall | Ein F1 | Ein Number | List Account Number Precision | List Account Number Recall | List Account Number F1 | List Account Number Number | Name Precision | Name Recall | Name F1 | Name Number | Ssn Precision | Ssn Recall | Ssn F1 | Ssn Number | Overall Precision | Overall Recall | Overall F1 | Overall Accuracy |
|:-------------:|:-----:|:----:|:---------------:|:-----------------:|:--------------:|:----------:|:--------------:|:-----------------------:|:--------------------:|:----------------:|:--------------------:|:-----------------------------:|:--------------------------:|:----------------------:|:--------------------------:|:-------------:|:----------:|:------:|:----------:|:-----------------------------:|:--------------------------:|:----------------------:|:--------------------------:|:--------------:|:-----------:|:-------:|:-----------:|:-------------:|:----------:|:------:|:----------:|:-----------------:|:--------------:|:----------:|:----------------:|
| 1.3523        | 1.0   | 234  | 0.7065          | 0.0               | 0.0            | 0.0        | 59             | 0.0                     | 0.0                  | 0.0              | 38                   | 0.0                           | 0.0                        | 0.0                    | 59                         | 0.0           | 0.0        | 0.0    | 20         | 0.0                           | 0.0                        | 0.0                    | 59                         | 0.0            | 0.0         | 0.0     | 59          | 0.0           | 0.0        | 0.0    | 39         | 0.0               | 0.0            | 0.0        | 0.9513           |
| 0.3676        | 2.0   | 468  | 0.1605          | 0.9667            | 0.9831         | 0.9748     | 59             | 0.9091                  | 0.7895               | 0.8451           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.0           | 0.0        | 0.0    | 20         | 0.6667                        | 0.8475                     | 0.7463                 | 59                         | 0.9077         | 1.0         | 0.9516  | 59          | 0.0           | 0.0        | 0.0    | 39         | 0.8767            | 0.7688         | 0.8192     | 0.9901           |
| 0.1217        | 3.0   | 702  | 0.0852          | 0.9667            | 0.9831         | 0.9748     | 59             | 0.9722                  | 0.9211               | 0.9459           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.0           | 0.0        | 0.0    | 20         | 0.7246                        | 0.8475                     | 0.7812                 | 59                         | 0.9833         | 1.0         | 0.9916  | 59          | 0.5574        | 0.8718     | 0.6800 | 39         | 0.8551            | 0.8859         | 0.8702     | 0.9953           |
| 0.0783        | 4.0   | 936  | 0.0590          | 1.0               | 1.0            | 1.0        | 59             | 0.9737                  | 0.9737               | 0.9737           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.0           | 0.0        | 0.0    | 20         | 0.9355                        | 0.9831                     | 0.9587                 | 59                         | 1.0            | 1.0         | 1.0     | 59          | 0.5161        | 0.8205     | 0.6337 | 39         | 0.8968            | 0.9129         | 0.9048     | 0.9959           |
| 0.0548        | 5.0   | 1170 | 0.0432          | 1.0               | 1.0            | 1.0        | 59             | 0.9737                  | 0.9737               | 0.9737           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.0           | 0.0        | 0.0    | 20         | 0.9667                        | 0.9831                     | 0.9748                 | 59                         | 1.0            | 1.0         | 1.0     | 59          | 0.55          | 0.8462     | 0.6667 | 39         | 0.9104            | 0.9159         | 0.9132     | 0.9963           |
| 0.0405        | 6.0   | 1404 | 0.0333          | 1.0               | 1.0            | 1.0        | 59             | 0.925                   | 0.9737               | 0.9487           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.0           | 0.0        | 0.0    | 20         | 0.9667                        | 0.9831                     | 0.9748                 | 59                         | 1.0            | 1.0         | 1.0     | 59          | 0.6066        | 0.9487     | 0.74   | 39         | 0.9142            | 0.9279         | 0.9210     | 0.9965           |
| 0.0328        | 7.0   | 1638 | 0.0278          | 0.9667            | 0.9831         | 0.9748     | 59             | 0.9737                  | 0.9737               | 0.9737           | 38                   | 0.9833                        | 1.0                        | 0.9916                 | 59                         | 0.0           | 0.0        | 0.0    | 20         | 1.0                           | 1.0                        | 1.0                    | 59                         | 1.0            | 1.0         | 1.0     | 59          | 0.5441        | 0.9487     | 0.6916 | 39         | 0.8983            | 0.9279         | 0.9129     | 0.9959           |
| 0.0245        | 8.0   | 1872 | 0.0212          | 1.0               | 1.0            | 1.0        | 59             | 0.9737                  | 0.9737               | 0.9737           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.1538        | 0.1        | 0.1212 | 20         | 0.9672                        | 1.0                        | 0.9833                 | 59                         | 1.0            | 1.0         | 1.0     | 59          | 0.5862        | 0.8718     | 0.7010 | 39         | 0.8905            | 0.9279         | 0.9088     | 0.9969           |
| 0.0192        | 9.0   | 2106 | 0.0164          | 1.0               | 1.0            | 1.0        | 59             | 0.9737                  | 0.9737               | 0.9737           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.56          | 0.7        | 0.6222 | 20         | 1.0                           | 1.0                        | 1.0                    | 59                         | 1.0            | 1.0         | 1.0     | 59          | 0.7111        | 0.8205     | 0.7619 | 39         | 0.9273            | 0.9580         | 0.9424     | 0.9983           |
| 0.0145        | 10.0  | 2340 | 0.0127          | 1.0               | 1.0            | 1.0        | 59             | 0.9737                  | 0.9737               | 0.9737           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.8235        | 0.7        | 0.7568 | 20         | 1.0                           | 1.0                        | 1.0                    | 59                         | 1.0            | 1.0         | 1.0     | 59          | 0.7391        | 0.8718     | 0.8000 | 39         | 0.9525            | 0.9640         | 0.9582     | 0.9989           |
| 0.0116        | 11.0  | 2574 | 0.0103          | 1.0               | 1.0            | 1.0        | 59             | 0.9737                  | 0.9737               | 0.9737           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.8571        | 0.9        | 0.8780 | 20         | 0.9672                        | 1.0                        | 0.9833                 | 59                         | 1.0            | 0.9661      | 0.9828  | 59          | 0.8537        | 0.8974     | 0.875  | 39         | 0.9643            | 0.9730         | 0.9686     | 0.9992           |
| 0.0099        | 12.0  | 2808 | 0.0095          | 1.0               | 1.0            | 1.0        | 59             | 0.9737                  | 0.9737               | 0.9737           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.9           | 0.9        | 0.9    | 20         | 1.0                           | 1.0                        | 1.0                    | 59                         | 1.0            | 1.0         | 1.0     | 59          | 0.8537        | 0.8974     | 0.875  | 39         | 0.9731            | 0.9790         | 0.9760     | 0.9992           |
| 0.0083        | 13.0  | 3042 | 0.0083          | 0.9667            | 0.9831         | 0.9748     | 59             | 0.9231                  | 0.9474               | 0.9351           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.8095        | 0.85       | 0.8293 | 20         | 0.9667                        | 0.9831                     | 0.9748                 | 59                         | 0.9667         | 0.9831      | 0.9748  | 59          | 0.875         | 0.8974     | 0.8861 | 39         | 0.9469            | 0.9640         | 0.9554     | 0.9990           |
| 0.0096        | 14.0  | 3276 | 0.0066          | 1.0               | 1.0            | 1.0        | 59             | 0.9231                  | 0.9474               | 0.9351           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.8571        | 0.9        | 0.8780 | 20         | 1.0                           | 1.0                        | 1.0                    | 59                         | 1.0            | 1.0         | 1.0     | 59          | 0.9024        | 0.9487     | 0.9250 | 39         | 0.9703            | 0.9820         | 0.9761     | 0.9993           |
| 0.0116        | 15.0  | 3510 | 0.0060          | 1.0               | 1.0            | 1.0        | 59             | 0.9737                  | 0.9737               | 0.9737           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.9048        | 0.95       | 0.9268 | 20         | 0.9667                        | 0.9831                     | 0.9748                 | 59                         | 1.0            | 1.0         | 1.0     | 59          | 0.8810        | 0.9487     | 0.9136 | 39         | 0.9704            | 0.9850         | 0.9776     | 0.9992           |
| 0.0064        | 16.0  | 3744 | 0.0045          | 1.0               | 1.0            | 1.0        | 59             | 0.9737                  | 0.9737               | 0.9737           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.8           | 0.8        | 0.8000 | 20         | 0.9833                        | 1.0                        | 0.9916                 | 59                         | 1.0            | 0.9831      | 0.9915  | 59          | 0.8837        | 0.9744     | 0.9268 | 39         | 0.9674            | 0.9790         | 0.9731     | 0.9995           |
| 0.0039        | 17.0  | 3978 | 0.0068          | 1.0               | 1.0            | 1.0        | 59             | 0.9737                  | 0.9737               | 0.9737           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 1.0           | 0.9        | 0.9474 | 20         | 0.9667                        | 0.9831                     | 0.9748                 | 59                         | 1.0            | 0.9661      | 0.9828  | 59          | 0.825         | 0.8462     | 0.8354 | 39         | 0.9698            | 0.9640         | 0.9669     | 0.9991           |
| 0.0036        | 18.0  | 4212 | 0.0098          | 1.0               | 1.0            | 1.0        | 59             | 0.9737                  | 0.9737               | 0.9737           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.5714        | 0.6        | 0.5854 | 20         | 0.9831                        | 0.9831                     | 0.9831                 | 59                         | 1.0            | 0.9831      | 0.9915  | 59          | 0.5424        | 0.8205     | 0.6531 | 39         | 0.8924            | 0.9459         | 0.9184     | 0.9981           |
| 0.0037        | 19.0  | 4446 | 0.0054          | 1.0               | 1.0            | 1.0        | 59             | 0.925                   | 0.9737               | 0.9487           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.9048        | 0.95       | 0.9268 | 20         | 0.9672                        | 1.0                        | 0.9833                 | 59                         | 0.9821         | 0.9322      | 0.9565  | 59          | 0.9231        | 0.9231     | 0.9231 | 39         | 0.9672            | 0.9730         | 0.9701     | 0.9991           |
| 0.0033        | 20.0  | 4680 | 0.0043          | 1.0               | 1.0            | 1.0        | 59             | 0.9737                  | 0.9737               | 0.9737           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.8182        | 0.9        | 0.8571 | 20         | 0.9672                        | 1.0                        | 0.9833                 | 59                         | 1.0            | 0.9661      | 0.9828  | 59          | 0.8810        | 0.9487     | 0.9136 | 39         | 0.9645            | 0.9790         | 0.9717     | 0.9992           |
| 0.0022        | 21.0  | 4914 | 0.0031          | 1.0               | 1.0            | 1.0        | 59             | 0.9737                  | 0.9737               | 0.9737           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.8571        | 0.9        | 0.8780 | 20         | 0.9833                        | 1.0                        | 0.9916                 | 59                         | 1.0            | 0.9831      | 0.9915  | 59          | 0.9048        | 0.9744     | 0.9383 | 39         | 0.9733            | 0.9850         | 0.9791     | 0.9995           |
| 0.0026        | 22.0  | 5148 | 0.0039          | 1.0               | 1.0            | 1.0        | 59             | 0.9737                  | 0.9737               | 0.9737           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 1.0           | 0.85       | 0.9189 | 20         | 1.0                           | 1.0                        | 1.0                    | 59                         | 1.0            | 1.0         | 1.0     | 59          | 0.8444        | 0.9744     | 0.9048 | 39         | 0.9762            | 0.9850         | 0.9806     | 0.9994           |
| 0.0018        | 23.0  | 5382 | 0.0026          | 1.0               | 1.0            | 1.0        | 59             | 0.9737                  | 0.9737               | 0.9737           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.8947        | 0.85       | 0.8718 | 20         | 1.0                           | 1.0                        | 1.0                    | 59                         | 1.0            | 1.0         | 1.0     | 59          | 0.9268        | 0.9744     | 0.9500 | 39         | 0.9820            | 0.9850         | 0.9835     | 0.9996           |
| 0.002         | 24.0  | 5616 | 0.0032          | 1.0               | 1.0            | 1.0        | 59             | 0.9737                  | 0.9737               | 0.9737           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.8571        | 0.9        | 0.8780 | 20         | 1.0                           | 1.0                        | 1.0                    | 59                         | 1.0            | 1.0         | 1.0     | 59          | 0.8605        | 0.9487     | 0.9024 | 39         | 0.9704            | 0.9850         | 0.9776     | 0.9995           |
| 0.0026        | 25.0  | 5850 | 0.0033          | 1.0               | 1.0            | 1.0        | 59             | 0.9737                  | 0.9737               | 0.9737           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.9048        | 0.95       | 0.9268 | 20         | 0.9672                        | 1.0                        | 0.9833                 | 59                         | 1.0            | 0.9661      | 0.9828  | 59          | 0.9048        | 0.9744     | 0.9383 | 39         | 0.9733            | 0.9850         | 0.9791     | 0.9994           |
| 0.0015        | 26.0  | 6084 | 0.0025          | 1.0               | 1.0            | 1.0        | 59             | 0.9737                  | 0.9737               | 0.9737           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.95          | 0.95       | 0.9500 | 20         | 0.9667                        | 0.9831                     | 0.9748                 | 59                         | 1.0            | 0.9831      | 0.9915  | 59          | 0.95          | 0.9744     | 0.9620 | 39         | 0.9820            | 0.9850         | 0.9835     | 0.9996           |
| 0.0022        | 27.0  | 6318 | 0.0029          | 1.0               | 1.0            | 1.0        | 59             | 0.9024                  | 0.9737               | 0.9367           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.8571        | 0.9        | 0.8780 | 20         | 1.0                           | 1.0                        | 1.0                    | 59                         | 1.0            | 1.0         | 1.0     | 59          | 0.9048        | 0.9744     | 0.9383 | 39         | 0.9676            | 0.9880         | 0.9777     | 0.9995           |
| 0.0012        | 28.0  | 6552 | 0.0031          | 1.0               | 1.0            | 1.0        | 59             | 0.9737                  | 0.9737               | 0.9737           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.9474        | 0.9        | 0.9231 | 20         | 1.0                           | 1.0                        | 1.0                    | 59                         | 1.0            | 1.0         | 1.0     | 59          | 0.9268        | 0.9744     | 0.9500 | 39         | 0.9850            | 0.9880         | 0.9865     | 0.9995           |
| 0.001         | 29.0  | 6786 | 0.0029          | 1.0               | 1.0            | 1.0        | 59             | 0.9737                  | 0.9737               | 0.9737           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.9444        | 0.85       | 0.8947 | 20         | 1.0                           | 1.0                        | 1.0                    | 59                         | 1.0            | 1.0         | 1.0     | 59          | 0.9048        | 0.9744     | 0.9383 | 39         | 0.9820            | 0.9850         | 0.9835     | 0.9995           |
| 0.0029        | 30.0  | 7020 | 0.0033          | 1.0               | 1.0            | 1.0        | 59             | 0.9737                  | 0.9737               | 0.9737           | 38                   | 1.0                           | 1.0                        | 1.0                    | 59                         | 0.95          | 0.95       | 0.9500 | 20         | 0.9667                        | 0.9831                     | 0.9748                 | 59                         | 1.0            | 1.0         | 1.0     | 59          | 0.95          | 0.9744     | 0.9620 | 39         | 0.9821            | 0.9880         | 0.9850     | 0.9995           |


### Framework versions

- Transformers 4.21.0.dev0
- Pytorch 1.11.0+cu113
- Datasets 2.2.2
- Tokenizers 0.12.1
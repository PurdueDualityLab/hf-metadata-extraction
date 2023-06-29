At its core it uses an ELECTRA-Base model (google/electra-base-discriminator) fine-tuned on the MS MARCO passage classification task. It can be loaded using the TF/AutoModelForSequenceClassification classes but it follows the same classification layer defined for BERT similarly to the TFElectraRelevanceHead in the Capreolus BERT-MaxP implementation.

Refer to our [github repository](https://github.com/BOUALILILila/ExactMatchMarking) for a usage example for ad hoc ranking.
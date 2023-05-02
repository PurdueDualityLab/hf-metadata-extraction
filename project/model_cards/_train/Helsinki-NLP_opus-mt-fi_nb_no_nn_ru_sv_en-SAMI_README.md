---
tags:
- translation
license: apache-2.0
---

### opus-mt-fi_nb_no_nn_ru_sv_en-SAMI

* source languages: fi,nb,no,nn,ru,sv,en
* target languages: se,sma,smj,smn,sms
*  OPUS readme: [fi+nb+no+nn+ru+sv+en-se+sma+smj+smn+sms](https://github.com/Helsinki-NLP/OPUS-MT-train/blob/master/models/fi+nb+no+nn+ru+sv+en-se+sma+smj+smn+sms/README.md)

*  dataset: opus+giella
* model: transformer-align
* pre-processing: normalization + SentencePiece
* a sentence initial language token is required in the form of `>>id<<` (id = valid target language ID)
* download original weights: [opus+giella-2020-04-18.zip](https://object.pouta.csc.fi/OPUS-MT-models/fi+nb+no+nn+ru+sv+en-se+sma+smj+smn+sms/opus+giella-2020-04-18.zip)
* test set translations: [opus+giella-2020-04-18.test.txt](https://object.pouta.csc.fi/OPUS-MT-models/fi+nb+no+nn+ru+sv+en-se+sma+smj+smn+sms/opus+giella-2020-04-18.test.txt)
* test set scores: [opus+giella-2020-04-18.eval.txt](https://object.pouta.csc.fi/OPUS-MT-models/fi+nb+no+nn+ru+sv+en-se+sma+smj+smn+sms/opus+giella-2020-04-18.eval.txt)

## Benchmarks

| testset               | BLEU  | chr-F |
|-----------------------|-------|-------|
| giella.fi.sms 	| 58.4 	| 0.776 |


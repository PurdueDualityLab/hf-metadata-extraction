---
tags:
- translation
license: apache-2.0
---

### opus-mt-en-mg

* source languages: en
* target languages: mg
*  OPUS readme: [en-mg](https://github.com/Helsinki-NLP/OPUS-MT-train/blob/master/models/en-mg/README.md)

*  dataset: opus
* model: transformer-align
* pre-processing: normalization + SentencePiece
* download original weights: [opus-2020-01-08.zip](https://object.pouta.csc.fi/OPUS-MT-models/en-mg/opus-2020-01-08.zip)
* test set translations: [opus-2020-01-08.test.txt](https://object.pouta.csc.fi/OPUS-MT-models/en-mg/opus-2020-01-08.test.txt)
* test set scores: [opus-2020-01-08.eval.txt](https://object.pouta.csc.fi/OPUS-MT-models/en-mg/opus-2020-01-08.eval.txt)

## Benchmarks

| testset               | BLEU  | chr-F |
|-----------------------|-------|-------|
| GlobalVoices.en.mg 	| 22.3 	| 0.565 |
| Tatoeba.en.mg 	| 35.5 	| 0.548 |


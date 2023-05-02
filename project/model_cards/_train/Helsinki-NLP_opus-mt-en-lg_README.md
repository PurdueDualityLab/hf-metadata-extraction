---
tags:
- translation
license: apache-2.0
---

### opus-mt-en-lg

* source languages: en
* target languages: lg
*  OPUS readme: [en-lg](https://github.com/Helsinki-NLP/OPUS-MT-train/blob/master/models/en-lg/README.md)

*  dataset: opus
* model: transformer-align
* pre-processing: normalization + SentencePiece
* download original weights: [opus-2020-01-08.zip](https://object.pouta.csc.fi/OPUS-MT-models/en-lg/opus-2020-01-08.zip)
* test set translations: [opus-2020-01-08.test.txt](https://object.pouta.csc.fi/OPUS-MT-models/en-lg/opus-2020-01-08.test.txt)
* test set scores: [opus-2020-01-08.eval.txt](https://object.pouta.csc.fi/OPUS-MT-models/en-lg/opus-2020-01-08.eval.txt)

## Benchmarks

| testset               | BLEU  | chr-F |
|-----------------------|-------|-------|
| JW300.en.lg 	| 30.4 	| 0.543 |
| Tatoeba.en.lg 	| 5.7 	| 0.386 |


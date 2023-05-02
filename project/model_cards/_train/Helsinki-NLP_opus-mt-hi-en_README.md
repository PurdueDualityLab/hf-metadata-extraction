---
tags:
- translation
license: apache-2.0
---

### opus-mt-hi-en

* source languages: hi
* target languages: en
*  OPUS readme: [hi-en](https://github.com/Helsinki-NLP/OPUS-MT-train/blob/master/models/hi-en/README.md)

*  dataset: opus
* model: transformer-align
* pre-processing: normalization + SentencePiece
* download original weights: [opus-2019-12-18.zip](https://object.pouta.csc.fi/OPUS-MT-models/hi-en/opus-2019-12-18.zip)
* test set translations: [opus-2019-12-18.test.txt](https://object.pouta.csc.fi/OPUS-MT-models/hi-en/opus-2019-12-18.test.txt)
* test set scores: [opus-2019-12-18.eval.txt](https://object.pouta.csc.fi/OPUS-MT-models/hi-en/opus-2019-12-18.eval.txt)

## Benchmarks

| testset               | BLEU  | chr-F |
|-----------------------|-------|-------|
| newsdev2014.hi.en 	| 9.1 	| 0.357 |
| newstest2014-hien.hi.en 	| 13.6 	| 0.409 |
| Tatoeba.hi.en 	| 40.4 	| 0.580 |


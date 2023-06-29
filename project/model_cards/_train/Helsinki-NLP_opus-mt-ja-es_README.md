---
tags:
- translation
license: apache-2.0
---

### opus-mt-ja-es

* source languages: ja
* target languages: es
*  OPUS readme: [ja-es](https://github.com/Helsinki-NLP/OPUS-MT-train/blob/master/models/ja-es/README.md)

*  dataset: opus
* model: transformer-align
* pre-processing: normalization + SentencePiece
* download original weights: [opus-2020-01-16.zip](https://object.pouta.csc.fi/OPUS-MT-models/ja-es/opus-2020-01-16.zip)
* test set translations: [opus-2020-01-16.test.txt](https://object.pouta.csc.fi/OPUS-MT-models/ja-es/opus-2020-01-16.test.txt)
* test set scores: [opus-2020-01-16.eval.txt](https://object.pouta.csc.fi/OPUS-MT-models/ja-es/opus-2020-01-16.eval.txt)

## Benchmarks

| testset               | BLEU  | chr-F |
|-----------------------|-------|-------|
| Tatoeba.ja.es 	| 34.6 	| 0.553 |

---
tags:
- translation
license: apache-2.0
---

### opus-mt-sv-fr

* source languages: sv
* target languages: fr
*  OPUS readme: [sv-fr](https://github.com/Helsinki-NLP/OPUS-MT-train/blob/master/models/sv-fr/README.md)

*  dataset: opus
* model: transformer-align
* pre-processing: normalization + SentencePiece
* download original weights: [opus-2020-01-24.zip](https://object.pouta.csc.fi/OPUS-MT-models/sv-fr/opus-2020-01-24.zip)
* test set translations: [opus-2020-01-24.test.txt](https://object.pouta.csc.fi/OPUS-MT-models/sv-fr/opus-2020-01-24.test.txt)
* test set scores: [opus-2020-01-24.eval.txt](https://object.pouta.csc.fi/OPUS-MT-models/sv-fr/opus-2020-01-24.eval.txt)

## Benchmarks

| testset               | BLEU  | chr-F |
|-----------------------|-------|-------|
| Tatoeba.sv.fr 	| 59.7 	| 0.731 |

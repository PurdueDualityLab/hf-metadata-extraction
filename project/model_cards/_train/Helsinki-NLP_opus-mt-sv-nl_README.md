---
tags:
- translation
license: apache-2.0
---

### opus-mt-sv-nl

* source languages: sv
* target languages: nl
*  OPUS readme: [sv-nl](https://github.com/Helsinki-NLP/OPUS-MT-train/blob/master/models/sv-nl/README.md)

*  dataset: opus
* model: transformer-align
* pre-processing: normalization + SentencePiece
* download original weights: [opus-2020-01-16.zip](https://object.pouta.csc.fi/OPUS-MT-models/sv-nl/opus-2020-01-16.zip)
* test set translations: [opus-2020-01-16.test.txt](https://object.pouta.csc.fi/OPUS-MT-models/sv-nl/opus-2020-01-16.test.txt)
* test set scores: [opus-2020-01-16.eval.txt](https://object.pouta.csc.fi/OPUS-MT-models/sv-nl/opus-2020-01-16.eval.txt)

## Benchmarks

| testset               | BLEU  | chr-F |
|-----------------------|-------|-------|
| GlobalVoices.sv.nl 	| 24.3 	| 0.522 |

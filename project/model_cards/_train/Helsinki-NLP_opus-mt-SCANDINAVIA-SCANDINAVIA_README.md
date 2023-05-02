---
tags:
- translation
license: apache-2.0
---

### opus-mt-SCANDINAVIA-SCANDINAVIA

* source languages: da,fo,is,no,nb,nn,sv
* target languages: da,fo,is,no,nb,nn,sv
*  OPUS readme: [da+fo+is+no+nb+nn+sv-da+fo+is+no+nb+nn+sv](https://github.com/Helsinki-NLP/OPUS-MT-train/blob/master/models/da+fo+is+no+nb+nn+sv-da+fo+is+no+nb+nn+sv/README.md)

*  dataset: opus
* model: transformer-align
* pre-processing: normalization + SentencePiece
* a sentence initial language token is required in the form of `>>id<<` (id = valid target language ID)
* download original weights: [opus-2019-12-18.zip](https://object.pouta.csc.fi/OPUS-MT-models/da+fo+is+no+nb+nn+sv-da+fo+is+no+nb+nn+sv/opus-2019-12-18.zip)
* test set translations: [opus-2019-12-18.test.txt](https://object.pouta.csc.fi/OPUS-MT-models/da+fo+is+no+nb+nn+sv-da+fo+is+no+nb+nn+sv/opus-2019-12-18.test.txt)
* test set scores: [opus-2019-12-18.eval.txt](https://object.pouta.csc.fi/OPUS-MT-models/da+fo+is+no+nb+nn+sv-da+fo+is+no+nb+nn+sv/opus-2019-12-18.eval.txt)

## Benchmarks

| testset               | BLEU  | chr-F |
|-----------------------|-------|-------|
| Tatoeba.da.sv 	| 69.2 	| 0.811 |


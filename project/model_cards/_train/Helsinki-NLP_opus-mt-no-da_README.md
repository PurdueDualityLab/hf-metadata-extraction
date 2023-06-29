---
language: 
- no
- da

tags:
- translation

license: apache-2.0
---

### nor-dan

* source group: Norwegian 
* target group: Danish 
*  OPUS readme: [nor-dan](https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/nor-dan/README.md)

*  model: transformer-align
* source language(s): nno nob
* target language(s): dan
* model: transformer-align
* pre-processing: normalization + SentencePiece (spm12k,spm12k)
* download original weights: [opus-2020-06-17.zip](https://object.pouta.csc.fi/Tatoeba-MT-models/nor-dan/opus-2020-06-17.zip)
* test set translations: [opus-2020-06-17.test.txt](https://object.pouta.csc.fi/Tatoeba-MT-models/nor-dan/opus-2020-06-17.test.txt)
* test set scores: [opus-2020-06-17.eval.txt](https://object.pouta.csc.fi/Tatoeba-MT-models/nor-dan/opus-2020-06-17.eval.txt)

## Benchmarks

| testset               | BLEU  | chr-F |
|-----------------------|-------|-------|
| Tatoeba-test.nor.dan 	| 65.0 	| 0.792 |


### System Info: 
- hf_name: nor-dan

- source_languages: nor

- target_languages: dan

- opus_readme_url: https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/nor-dan/README.md

- original_repo: Tatoeba-Challenge

- tags: ['translation']

- languages: ['no', 'da']

- src_constituents: {'nob', 'nno'}

- tgt_constituents: {'dan'}

- src_multilingual: False

- tgt_multilingual: False

- prepro:  normalization + SentencePiece (spm12k,spm12k)

- url_model: https://object.pouta.csc.fi/Tatoeba-MT-models/nor-dan/opus-2020-06-17.zip

- url_test_set: https://object.pouta.csc.fi/Tatoeba-MT-models/nor-dan/opus-2020-06-17.test.txt

- src_alpha3: nor

- tgt_alpha3: dan

- short_pair: no-da

- chrF2_score: 0.792

- bleu: 65.0

- brevity_penalty: 0.995

- ref_len: 9865.0

- src_name: Norwegian

- tgt_name: Danish

- train_date: 2020-06-17

- src_alpha2: no

- tgt_alpha2: da

- prefer_old: False

- long_pair: nor-dan

- helsinki_git_sha: 480fcbe0ee1bf4774bcbe6226ad9f58e63f6c535

- transformers_git_sha: 2207e5d8cb224e954a7cba69fa4ac2309e9ff30b

- port_machine: brutasse

- port_time: 2020-08-21-14:41
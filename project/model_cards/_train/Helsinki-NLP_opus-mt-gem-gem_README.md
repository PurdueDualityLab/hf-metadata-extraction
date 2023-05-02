---
language: 
- da
- sv
- af
- nn
- fy
- fo
- de
- nb
- nl
- is
- en
- lb
- yi
- gem

tags:
- translation

license: apache-2.0
---

### gem-gem

* source group: Germanic languages 
* target group: Germanic languages 
*  OPUS readme: [gem-gem](https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/gem-gem/README.md)

*  model: transformer
* source language(s): afr ang_Latn dan deu eng enm_Latn fao frr fry gos got_Goth gsw isl ksh ltz nds nld nno nob nob_Hebr non_Latn pdc sco stq swe swg yid
* target language(s): afr ang_Latn dan deu eng enm_Latn fao frr fry gos got_Goth gsw isl ksh ltz nds nld nno nob nob_Hebr non_Latn pdc sco stq swe swg yid
* model: transformer
* pre-processing: normalization + SentencePiece (spm32k,spm32k)
* a sentence initial language token is required in the form of `>>id<<` (id = valid target language ID)
* download original weights: [opus-2020-07-27.zip](https://object.pouta.csc.fi/Tatoeba-MT-models/gem-gem/opus-2020-07-27.zip)
* test set translations: [opus-2020-07-27.test.txt](https://object.pouta.csc.fi/Tatoeba-MT-models/gem-gem/opus-2020-07-27.test.txt)
* test set scores: [opus-2020-07-27.eval.txt](https://object.pouta.csc.fi/Tatoeba-MT-models/gem-gem/opus-2020-07-27.eval.txt)

## Benchmarks

| testset               | BLEU  | chr-F |
|-----------------------|-------|-------|
| newssyscomb2009-deueng.deu.eng 	| 24.5 	| 0.519 |
| newssyscomb2009-engdeu.eng.deu 	| 18.7 	| 0.495 |
| news-test2008-deueng.deu.eng 	| 22.8 	| 0.509 |
| news-test2008-engdeu.eng.deu 	| 18.6 	| 0.485 |
| newstest2009-deueng.deu.eng 	| 22.2 	| 0.507 |
| newstest2009-engdeu.eng.deu 	| 18.3 	| 0.491 |
| newstest2010-deueng.deu.eng 	| 24.8 	| 0.537 |
| newstest2010-engdeu.eng.deu 	| 19.7 	| 0.499 |
| newstest2011-deueng.deu.eng 	| 22.9 	| 0.516 |
| newstest2011-engdeu.eng.deu 	| 18.3 	| 0.485 |
| newstest2012-deueng.deu.eng 	| 23.9 	| 0.524 |
| newstest2012-engdeu.eng.deu 	| 18.5 	| 0.484 |
| newstest2013-deueng.deu.eng 	| 26.3 	| 0.537 |
| newstest2013-engdeu.eng.deu 	| 21.5 	| 0.506 |
| newstest2014-deen-deueng.deu.eng 	| 25.7 	| 0.535 |
| newstest2015-ende-deueng.deu.eng 	| 27.3 	| 0.542 |
| newstest2015-ende-engdeu.eng.deu 	| 24.2 	| 0.534 |
| newstest2016-ende-deueng.deu.eng 	| 31.8 	| 0.584 |
| newstest2016-ende-engdeu.eng.deu 	| 28.4 	| 0.564 |
| newstest2017-ende-deueng.deu.eng 	| 27.6 	| 0.545 |
| newstest2017-ende-engdeu.eng.deu 	| 22.8 	| 0.527 |
| newstest2018-ende-deueng.deu.eng 	| 34.1 	| 0.593 |
| newstest2018-ende-engdeu.eng.deu 	| 32.7 	| 0.595 |
| newstest2019-deen-deueng.deu.eng 	| 30.6 	| 0.565 |
| newstest2019-ende-engdeu.eng.deu 	| 29.5 	| 0.567 |
| Tatoeba-test.afr-ang.afr.ang 	| 0.0 	| 0.053 |
| Tatoeba-test.afr-dan.afr.dan 	| 57.8 	| 0.907 |
| Tatoeba-test.afr-deu.afr.deu 	| 46.4 	| 0.663 |
| Tatoeba-test.afr-eng.afr.eng 	| 57.4 	| 0.717 |
| Tatoeba-test.afr-enm.afr.enm 	| 11.3 	| 0.285 |
| Tatoeba-test.afr-fry.afr.fry 	| 0.0 	| 0.167 |
| Tatoeba-test.afr-gos.afr.gos 	| 1.5 	| 0.178 |
| Tatoeba-test.afr-isl.afr.isl 	| 29.0 	| 0.760 |
| Tatoeba-test.afr-ltz.afr.ltz 	| 11.2 	| 0.246 |
| Tatoeba-test.afr-nld.afr.nld 	| 53.3 	| 0.708 |
| Tatoeba-test.afr-nor.afr.nor 	| 66.0 	| 0.752 |
| Tatoeba-test.afr-swe.afr.swe 	| 88.0 	| 0.955 |
| Tatoeba-test.afr-yid.afr.yid 	| 59.5 	| 0.443 |
| Tatoeba-test.ang-afr.ang.afr 	| 10.7 	| 0.043 |
| Tatoeba-test.ang-dan.ang.dan 	| 6.3 	| 0.190 |
| Tatoeba-test.ang-deu.ang.deu 	| 1.4 	| 0.212 |
| Tatoeba-test.ang-eng.ang.eng 	| 8.1 	| 0.247 |
| Tatoeba-test.ang-enm.ang.enm 	| 1.7 	| 0.196 |
| Tatoeba-test.ang-fao.ang.fao 	| 10.7 	| 0.105 |
| Tatoeba-test.ang-gos.ang.gos 	| 10.7 	| 0.128 |
| Tatoeba-test.ang-isl.ang.isl 	| 16.0 	| 0.135 |
| Tatoeba-test.ang-ltz.ang.ltz 	| 16.0 	| 0.121 |
| Tatoeba-test.ang-yid.ang.yid 	| 1.5 	| 0.136 |
| Tatoeba-test.dan-afr.dan.afr 	| 22.7 	| 0.655 |
| Tatoeba-test.dan-ang.dan.ang 	| 3.1 	| 0.110 |
| Tatoeba-test.dan-deu.dan.deu 	| 47.4 	| 0.676 |
| Tatoeba-test.dan-eng.dan.eng 	| 54.7 	| 0.704 |
| Tatoeba-test.dan-enm.dan.enm 	| 4.8 	| 0.291 |
| Tatoeba-test.dan-fao.dan.fao 	| 9.7 	| 0.120 |
| Tatoeba-test.dan-gos.dan.gos 	| 3.8 	| 0.240 |
| Tatoeba-test.dan-isl.dan.isl 	| 66.1 	| 0.678 |
| Tatoeba-test.dan-ltz.dan.ltz 	| 78.3 	| 0.563 |
| Tatoeba-test.dan-nds.dan.nds 	| 6.2 	| 0.335 |
| Tatoeba-test.dan-nld.dan.nld 	| 60.0 	| 0.748 |
| Tatoeba-test.dan-nor.dan.nor 	| 68.1 	| 0.812 |
| Tatoeba-test.dan-swe.dan.swe 	| 65.0 	| 0.785 |
| Tatoeba-test.dan-swg.dan.swg 	| 2.6 	| 0.182 |
| Tatoeba-test.dan-yid.dan.yid 	| 9.3 	| 0.226 |
| Tatoeba-test.deu-afr.deu.afr 	| 50.3 	| 0.682 |
| Tatoeba-test.deu-ang.deu.ang 	| 0.5 	| 0.118 |
| Tatoeba-test.deu-dan.deu.dan 	| 49.6 	| 0.679 |
| Tatoeba-test.deu-eng.deu.eng 	| 43.4 	| 0.618 |
| Tatoeba-test.deu-enm.deu.enm 	| 2.2 	| 0.159 |
| Tatoeba-test.deu-frr.deu.frr 	| 0.4 	| 0.156 |
| Tatoeba-test.deu-fry.deu.fry 	| 10.7 	| 0.355 |
| Tatoeba-test.deu-gos.deu.gos 	| 0.7 	| 0.183 |
| Tatoeba-test.deu-got.deu.got 	| 0.3 	| 0.010 |
| Tatoeba-test.deu-gsw.deu.gsw 	| 1.1 	| 0.130 |
| Tatoeba-test.deu-isl.deu.isl 	| 24.3 	| 0.504 |
| Tatoeba-test.deu-ksh.deu.ksh 	| 0.9 	| 0.173 |
| Tatoeba-test.deu-ltz.deu.ltz 	| 15.6 	| 0.304 |
| Tatoeba-test.deu-nds.deu.nds 	| 21.2 	| 0.469 |
| Tatoeba-test.deu-nld.deu.nld 	| 47.1 	| 0.657 |
| Tatoeba-test.deu-nor.deu.nor 	| 43.9 	| 0.646 |
| Tatoeba-test.deu-pdc.deu.pdc 	| 3.0 	| 0.133 |
| Tatoeba-test.deu-sco.deu.sco 	| 12.0 	| 0.296 |
| Tatoeba-test.deu-stq.deu.stq 	| 0.6 	| 0.137 |
| Tatoeba-test.deu-swe.deu.swe 	| 50.6 	| 0.668 |
| Tatoeba-test.deu-swg.deu.swg 	| 0.2 	| 0.137 |
| Tatoeba-test.deu-yid.deu.yid 	| 3.9 	| 0.229 |
| Tatoeba-test.eng-afr.eng.afr 	| 55.2 	| 0.721 |
| Tatoeba-test.eng-ang.eng.ang 	| 4.9 	| 0.118 |
| Tatoeba-test.eng-dan.eng.dan 	| 52.6 	| 0.684 |
| Tatoeba-test.eng-deu.eng.deu 	| 35.4 	| 0.573 |
| Tatoeba-test.eng-enm.eng.enm 	| 1.8 	| 0.223 |
| Tatoeba-test.eng-fao.eng.fao 	| 7.0 	| 0.312 |
| Tatoeba-test.eng-frr.eng.frr 	| 1.2 	| 0.050 |
| Tatoeba-test.eng-fry.eng.fry 	| 15.8 	| 0.381 |
| Tatoeba-test.eng-gos.eng.gos 	| 0.7 	| 0.170 |
| Tatoeba-test.eng-got.eng.got 	| 0.3 	| 0.011 |
| Tatoeba-test.eng-gsw.eng.gsw 	| 0.5 	| 0.126 |
| Tatoeba-test.eng-isl.eng.isl 	| 20.9 	| 0.463 |
| Tatoeba-test.eng-ksh.eng.ksh 	| 1.0 	| 0.141 |
| Tatoeba-test.eng-ltz.eng.ltz 	| 12.8 	| 0.292 |
| Tatoeba-test.eng-nds.eng.nds 	| 18.3 	| 0.428 |
| Tatoeba-test.eng-nld.eng.nld 	| 47.3 	| 0.657 |
| Tatoeba-test.eng-non.eng.non 	| 0.3 	| 0.145 |
| Tatoeba-test.eng-nor.eng.nor 	| 47.2 	| 0.650 |
| Tatoeba-test.eng-pdc.eng.pdc 	| 4.8 	| 0.177 |
| Tatoeba-test.eng-sco.eng.sco 	| 38.1 	| 0.597 |
| Tatoeba-test.eng-stq.eng.stq 	| 2.4 	| 0.288 |
| Tatoeba-test.eng-swe.eng.swe 	| 52.7 	| 0.677 |
| Tatoeba-test.eng-swg.eng.swg 	| 1.1 	| 0.163 |
| Tatoeba-test.eng-yid.eng.yid 	| 4.5 	| 0.223 |
| Tatoeba-test.enm-afr.enm.afr 	| 22.8 	| 0.401 |
| Tatoeba-test.enm-ang.enm.ang 	| 0.4 	| 0.062 |
| Tatoeba-test.enm-dan.enm.dan 	| 51.4 	| 0.782 |
| Tatoeba-test.enm-deu.enm.deu 	| 33.8 	| 0.473 |
| Tatoeba-test.enm-eng.enm.eng 	| 22.4 	| 0.495 |
| Tatoeba-test.enm-fry.enm.fry 	| 16.0 	| 0.173 |
| Tatoeba-test.enm-gos.enm.gos 	| 6.1 	| 0.222 |
| Tatoeba-test.enm-isl.enm.isl 	| 59.5 	| 0.651 |
| Tatoeba-test.enm-ksh.enm.ksh 	| 10.5 	| 0.130 |
| Tatoeba-test.enm-nds.enm.nds 	| 18.1 	| 0.327 |
| Tatoeba-test.enm-nld.enm.nld 	| 38.3 	| 0.546 |
| Tatoeba-test.enm-nor.enm.nor 	| 15.6 	| 0.290 |
| Tatoeba-test.enm-yid.enm.yid 	| 2.3 	| 0.215 |
| Tatoeba-test.fao-ang.fao.ang 	| 2.1 	| 0.035 |
| Tatoeba-test.fao-dan.fao.dan 	| 53.7 	| 0.625 |
| Tatoeba-test.fao-eng.fao.eng 	| 24.7 	| 0.435 |
| Tatoeba-test.fao-gos.fao.gos 	| 12.7 	| 0.116 |
| Tatoeba-test.fao-isl.fao.isl 	| 26.3 	| 0.341 |
| Tatoeba-test.fao-nor.fao.nor 	| 41.9 	| 0.586 |
| Tatoeba-test.fao-swe.fao.swe 	| 0.0 	| 1.000 |
| Tatoeba-test.frr-deu.frr.deu 	| 7.4 	| 0.263 |
| Tatoeba-test.frr-eng.frr.eng 	| 7.0 	| 0.157 |
| Tatoeba-test.frr-fry.frr.fry 	| 4.0 	| 0.112 |
| Tatoeba-test.frr-gos.frr.gos 	| 1.0 	| 0.135 |
| Tatoeba-test.frr-nds.frr.nds 	| 12.4 	| 0.207 |
| Tatoeba-test.frr-nld.frr.nld 	| 10.6 	| 0.227 |
| Tatoeba-test.frr-stq.frr.stq 	| 1.0 	| 0.058 |
| Tatoeba-test.fry-afr.fry.afr 	| 12.7 	| 0.333 |
| Tatoeba-test.fry-deu.fry.deu 	| 30.8 	| 0.555 |
| Tatoeba-test.fry-eng.fry.eng 	| 31.2 	| 0.506 |
| Tatoeba-test.fry-enm.fry.enm 	| 0.0 	| 0.175 |
| Tatoeba-test.fry-frr.fry.frr 	| 1.6 	| 0.091 |
| Tatoeba-test.fry-gos.fry.gos 	| 1.1 	| 0.254 |
| Tatoeba-test.fry-ltz.fry.ltz 	| 30.4 	| 0.526 |
| Tatoeba-test.fry-nds.fry.nds 	| 12.4 	| 0.116 |
| Tatoeba-test.fry-nld.fry.nld 	| 43.4 	| 0.637 |
| Tatoeba-test.fry-nor.fry.nor 	| 47.1 	| 0.607 |
| Tatoeba-test.fry-stq.fry.stq 	| 0.6 	| 0.181 |
| Tatoeba-test.fry-swe.fry.swe 	| 30.2 	| 0.587 |
| Tatoeba-test.fry-yid.fry.yid 	| 3.1 	| 0.173 |
| Tatoeba-test.gos-afr.gos.afr 	| 1.8 	| 0.215 |
| Tatoeba-test.gos-ang.gos.ang 	| 0.0 	| 0.045 |
| Tatoeba-test.gos-dan.gos.dan 	| 4.1 	| 0.236 |
| Tatoeba-test.gos-deu.gos.deu 	| 19.6 	| 0.406 |
| Tatoeba-test.gos-eng.gos.eng 	| 15.1 	| 0.329 |
| Tatoeba-test.gos-enm.gos.enm 	| 5.8 	| 0.271 |
| Tatoeba-test.gos-fao.gos.fao 	| 19.0 	| 0.136 |
| Tatoeba-test.gos-frr.gos.frr 	| 1.3 	| 0.119 |
| Tatoeba-test.gos-fry.gos.fry 	| 17.1 	| 0.388 |
| Tatoeba-test.gos-isl.gos.isl 	| 16.8 	| 0.356 |
| Tatoeba-test.gos-ltz.gos.ltz 	| 3.6 	| 0.174 |
| Tatoeba-test.gos-nds.gos.nds 	| 4.7 	| 0.225 |
| Tatoeba-test.gos-nld.gos.nld 	| 16.3 	| 0.406 |
| Tatoeba-test.gos-stq.gos.stq 	| 0.7 	| 0.154 |
| Tatoeba-test.gos-swe.gos.swe 	| 8.6 	| 0.319 |
| Tatoeba-test.gos-yid.gos.yid 	| 4.4 	| 0.165 |
| Tatoeba-test.got-deu.got.deu 	| 0.2 	| 0.041 |
| Tatoeba-test.got-eng.got.eng 	| 0.2 	| 0.068 |
| Tatoeba-test.got-nor.got.nor 	| 0.6 	| 0.000 |
| Tatoeba-test.gsw-deu.gsw.deu 	| 15.9 	| 0.373 |
| Tatoeba-test.gsw-eng.gsw.eng 	| 14.7 	| 0.320 |
| Tatoeba-test.isl-afr.isl.afr 	| 38.0 	| 0.641 |
| Tatoeba-test.isl-ang.isl.ang 	| 0.0 	| 0.037 |
| Tatoeba-test.isl-dan.isl.dan 	| 67.7 	| 0.836 |
| Tatoeba-test.isl-deu.isl.deu 	| 42.6 	| 0.614 |
| Tatoeba-test.isl-eng.isl.eng 	| 43.5 	| 0.610 |
| Tatoeba-test.isl-enm.isl.enm 	| 12.4 	| 0.123 |
| Tatoeba-test.isl-fao.isl.fao 	| 15.6 	| 0.176 |
| Tatoeba-test.isl-gos.isl.gos 	| 7.1 	| 0.257 |
| Tatoeba-test.isl-nor.isl.nor 	| 53.5 	| 0.690 |
| Tatoeba-test.isl-stq.isl.stq 	| 10.7 	| 0.176 |
| Tatoeba-test.isl-swe.isl.swe 	| 67.7 	| 0.818 |
| Tatoeba-test.ksh-deu.ksh.deu 	| 11.8 	| 0.393 |
| Tatoeba-test.ksh-eng.ksh.eng 	| 4.0 	| 0.239 |
| Tatoeba-test.ksh-enm.ksh.enm 	| 9.5 	| 0.085 |
| Tatoeba-test.ltz-afr.ltz.afr 	| 36.5 	| 0.529 |
| Tatoeba-test.ltz-ang.ltz.ang 	| 0.0 	| 0.043 |
| Tatoeba-test.ltz-dan.ltz.dan 	| 80.6 	| 0.722 |
| Tatoeba-test.ltz-deu.ltz.deu 	| 40.1 	| 0.581 |
| Tatoeba-test.ltz-eng.ltz.eng 	| 36.1 	| 0.511 |
| Tatoeba-test.ltz-fry.ltz.fry 	| 16.5 	| 0.524 |
| Tatoeba-test.ltz-gos.ltz.gos 	| 0.7 	| 0.118 |
| Tatoeba-test.ltz-nld.ltz.nld 	| 40.4 	| 0.535 |
| Tatoeba-test.ltz-nor.ltz.nor 	| 19.1 	| 0.582 |
| Tatoeba-test.ltz-stq.ltz.stq 	| 2.4 	| 0.093 |
| Tatoeba-test.ltz-swe.ltz.swe 	| 25.9 	| 0.430 |
| Tatoeba-test.ltz-yid.ltz.yid 	| 1.5 	| 0.160 |
| Tatoeba-test.multi.multi 	| 42.7 	| 0.614 |
| Tatoeba-test.nds-dan.nds.dan 	| 23.0 	| 0.465 |
| Tatoeba-test.nds-deu.nds.deu 	| 39.8 	| 0.610 |
| Tatoeba-test.nds-eng.nds.eng 	| 32.0 	| 0.520 |
| Tatoeba-test.nds-enm.nds.enm 	| 3.9 	| 0.156 |
| Tatoeba-test.nds-frr.nds.frr 	| 10.7 	| 0.127 |
| Tatoeba-test.nds-fry.nds.fry 	| 10.7 	| 0.231 |
| Tatoeba-test.nds-gos.nds.gos 	| 0.8 	| 0.157 |
| Tatoeba-test.nds-nld.nds.nld 	| 44.1 	| 0.634 |
| Tatoeba-test.nds-nor.nds.nor 	| 47.1 	| 0.665 |
| Tatoeba-test.nds-swg.nds.swg 	| 0.5 	| 0.166 |
| Tatoeba-test.nds-yid.nds.yid 	| 12.7 	| 0.337 |
| Tatoeba-test.nld-afr.nld.afr 	| 58.4 	| 0.748 |
| Tatoeba-test.nld-dan.nld.dan 	| 61.3 	| 0.753 |
| Tatoeba-test.nld-deu.nld.deu 	| 48.2 	| 0.670 |
| Tatoeba-test.nld-eng.nld.eng 	| 52.8 	| 0.690 |
| Tatoeba-test.nld-enm.nld.enm 	| 5.7 	| 0.178 |
| Tatoeba-test.nld-frr.nld.frr 	| 0.9 	| 0.159 |
| Tatoeba-test.nld-fry.nld.fry 	| 23.0 	| 0.467 |
| Tatoeba-test.nld-gos.nld.gos 	| 1.0 	| 0.165 |
| Tatoeba-test.nld-ltz.nld.ltz 	| 14.4 	| 0.310 |
| Tatoeba-test.nld-nds.nld.nds 	| 24.1 	| 0.485 |
| Tatoeba-test.nld-nor.nld.nor 	| 53.6 	| 0.705 |
| Tatoeba-test.nld-sco.nld.sco 	| 15.0 	| 0.415 |
| Tatoeba-test.nld-stq.nld.stq 	| 0.5 	| 0.183 |
| Tatoeba-test.nld-swe.nld.swe 	| 73.6 	| 0.842 |
| Tatoeba-test.nld-swg.nld.swg 	| 4.2 	| 0.191 |
| Tatoeba-test.nld-yid.nld.yid 	| 9.4 	| 0.299 |
| Tatoeba-test.non-eng.non.eng 	| 27.7 	| 0.501 |
| Tatoeba-test.nor-afr.nor.afr 	| 48.2 	| 0.687 |
| Tatoeba-test.nor-dan.nor.dan 	| 69.5 	| 0.820 |
| Tatoeba-test.nor-deu.nor.deu 	| 41.1 	| 0.634 |
| Tatoeba-test.nor-eng.nor.eng 	| 49.4 	| 0.660 |
| Tatoeba-test.nor-enm.nor.enm 	| 6.8 	| 0.230 |
| Tatoeba-test.nor-fao.nor.fao 	| 6.9 	| 0.395 |
| Tatoeba-test.nor-fry.nor.fry 	| 9.2 	| 0.323 |
| Tatoeba-test.nor-got.nor.got 	| 1.5 	| 0.000 |
| Tatoeba-test.nor-isl.nor.isl 	| 34.5 	| 0.555 |
| Tatoeba-test.nor-ltz.nor.ltz 	| 22.1 	| 0.447 |
| Tatoeba-test.nor-nds.nor.nds 	| 34.3 	| 0.565 |
| Tatoeba-test.nor-nld.nor.nld 	| 50.5 	| 0.676 |
| Tatoeba-test.nor-nor.nor.nor 	| 57.6 	| 0.764 |
| Tatoeba-test.nor-swe.nor.swe 	| 68.9 	| 0.813 |
| Tatoeba-test.nor-yid.nor.yid 	| 65.0 	| 0.627 |
| Tatoeba-test.pdc-deu.pdc.deu 	| 43.5 	| 0.559 |
| Tatoeba-test.pdc-eng.pdc.eng 	| 26.1 	| 0.471 |
| Tatoeba-test.sco-deu.sco.deu 	| 7.1 	| 0.295 |
| Tatoeba-test.sco-eng.sco.eng 	| 34.4 	| 0.551 |
| Tatoeba-test.sco-nld.sco.nld 	| 9.9 	| 0.438 |
| Tatoeba-test.stq-deu.stq.deu 	| 8.6 	| 0.385 |
| Tatoeba-test.stq-eng.stq.eng 	| 21.8 	| 0.431 |
| Tatoeba-test.stq-frr.stq.frr 	| 2.1 	| 0.111 |
| Tatoeba-test.stq-fry.stq.fry 	| 7.6 	| 0.267 |
| Tatoeba-test.stq-gos.stq.gos 	| 0.7 	| 0.198 |
| Tatoeba-test.stq-isl.stq.isl 	| 16.0 	| 0.121 |
| Tatoeba-test.stq-ltz.stq.ltz 	| 3.8 	| 0.150 |
| Tatoeba-test.stq-nld.stq.nld 	| 14.6 	| 0.375 |
| Tatoeba-test.stq-yid.stq.yid 	| 2.4 	| 0.096 |
| Tatoeba-test.swe-afr.swe.afr 	| 51.8 	| 0.802 |
| Tatoeba-test.swe-dan.swe.dan 	| 64.9 	| 0.784 |
| Tatoeba-test.swe-deu.swe.deu 	| 47.0 	| 0.657 |
| Tatoeba-test.swe-eng.swe.eng 	| 55.8 	| 0.700 |
| Tatoeba-test.swe-fao.swe.fao 	| 0.0 	| 0.060 |
| Tatoeba-test.swe-fry.swe.fry 	| 14.1 	| 0.449 |
| Tatoeba-test.swe-gos.swe.gos 	| 7.5 	| 0.291 |
| Tatoeba-test.swe-isl.swe.isl 	| 70.7 	| 0.812 |
| Tatoeba-test.swe-ltz.swe.ltz 	| 15.9 	| 0.553 |
| Tatoeba-test.swe-nld.swe.nld 	| 78.7 	| 0.854 |
| Tatoeba-test.swe-nor.swe.nor 	| 67.1 	| 0.799 |
| Tatoeba-test.swe-yid.swe.yid 	| 14.7 	| 0.156 |
| Tatoeba-test.swg-dan.swg.dan 	| 7.7 	| 0.341 |
| Tatoeba-test.swg-deu.swg.deu 	| 8.0 	| 0.334 |
| Tatoeba-test.swg-eng.swg.eng 	| 12.4 	| 0.305 |
| Tatoeba-test.swg-nds.swg.nds 	| 1.1 	| 0.209 |
| Tatoeba-test.swg-nld.swg.nld 	| 4.9 	| 0.244 |
| Tatoeba-test.swg-yid.swg.yid 	| 3.4 	| 0.194 |
| Tatoeba-test.yid-afr.yid.afr 	| 23.6 	| 0.552 |
| Tatoeba-test.yid-ang.yid.ang 	| 0.1 	| 0.066 |
| Tatoeba-test.yid-dan.yid.dan 	| 17.5 	| 0.392 |
| Tatoeba-test.yid-deu.yid.deu 	| 21.0 	| 0.423 |
| Tatoeba-test.yid-eng.yid.eng 	| 17.4 	| 0.368 |
| Tatoeba-test.yid-enm.yid.enm 	| 0.6 	| 0.143 |
| Tatoeba-test.yid-fry.yid.fry 	| 5.3 	| 0.169 |
| Tatoeba-test.yid-gos.yid.gos 	| 1.2 	| 0.149 |
| Tatoeba-test.yid-ltz.yid.ltz 	| 3.5 	| 0.256 |
| Tatoeba-test.yid-nds.yid.nds 	| 14.4 	| 0.487 |
| Tatoeba-test.yid-nld.yid.nld 	| 26.1 	| 0.423 |
| Tatoeba-test.yid-nor.yid.nor 	| 47.1 	| 0.583 |
| Tatoeba-test.yid-stq.yid.stq 	| 1.5 	| 0.092 |
| Tatoeba-test.yid-swe.yid.swe 	| 35.9 	| 0.518 |
| Tatoeba-test.yid-swg.yid.swg 	| 1.0 	| 0.124 |


### System Info: 
- hf_name: gem-gem

- source_languages: gem

- target_languages: gem

- opus_readme_url: https://github.com/Helsinki-NLP/Tatoeba-Challenge/tree/master/models/gem-gem/README.md

- original_repo: Tatoeba-Challenge

- tags: ['translation']

- languages: ['da', 'sv', 'af', 'nn', 'fy', 'fo', 'de', 'nb', 'nl', 'is', 'en', 'lb', 'yi', 'gem']

- src_constituents: {'ksh', 'enm_Latn', 'got_Goth', 'stq', 'dan', 'swe', 'afr', 'pdc', 'gos', 'nno', 'fry', 'gsw', 'fao', 'deu', 'swg', 'sco', 'nob', 'nld', 'isl', 'eng', 'ltz', 'nob_Hebr', 'ang_Latn', 'frr', 'non_Latn', 'yid', 'nds'}

- tgt_constituents: {'ksh', 'enm_Latn', 'got_Goth', 'stq', 'dan', 'swe', 'afr', 'pdc', 'gos', 'nno', 'fry', 'gsw', 'fao', 'deu', 'swg', 'sco', 'nob', 'nld', 'isl', 'eng', 'ltz', 'nob_Hebr', 'ang_Latn', 'frr', 'non_Latn', 'yid', 'nds'}

- src_multilingual: True

- tgt_multilingual: True

- prepro:  normalization + SentencePiece (spm32k,spm32k)

- url_model: https://object.pouta.csc.fi/Tatoeba-MT-models/gem-gem/opus-2020-07-27.zip

- url_test_set: https://object.pouta.csc.fi/Tatoeba-MT-models/gem-gem/opus-2020-07-27.test.txt

- src_alpha3: gem

- tgt_alpha3: gem

- short_pair: gem-gem

- chrF2_score: 0.614

- bleu: 42.7

- brevity_penalty: 0.993

- ref_len: 73459.0

- src_name: Germanic languages

- tgt_name: Germanic languages

- train_date: 2020-07-27

- src_alpha2: gem

- tgt_alpha2: gem

- prefer_old: False

- long_pair: gem-gem

- helsinki_git_sha: 480fcbe0ee1bf4774bcbe6226ad9f58e63f6c535

- transformers_git_sha: 2207e5d8cb224e954a7cba69fa4ac2309e9ff30b

- port_machine: brutasse

- port_time: 2020-08-21-14:41
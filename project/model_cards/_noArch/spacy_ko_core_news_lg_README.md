---
tags:
- spacy
- token-classification
language:
- ko
license: cc-by-sa-4.0
model-index:
- name: ko_core_news_lg
  results:
  - task:
      name: NER
      type: token-classification
    metrics:
    - name: NER Precision
      type: precision
      value: 0.8669446273
    - name: NER Recall
      type: recall
      value: 0.837301307
    - name: NER F Score
      type: f_score
      value: 0.8518651621
  - task:
      name: TAG
      type: token-classification
    metrics:
    - name: TAG (XPOS) Accuracy
      type: accuracy
      value: 0.8400253175
  - task:
      name: POS
      type: token-classification
    metrics:
    - name: POS (UPOS) Accuracy
      type: accuracy
      value: 0.9487717077
  - task:
      name: LEMMA
      type: token-classification
    metrics:
    - name: Lemma Accuracy
      type: accuracy
      value: 0.9009276291
  - task:
      name: UNLABELED_DEPENDENCIES
      type: token-classification
    metrics:
    - name: Unlabeled Attachment Score (UAS)
      type: f_score
      value: 0.8416620252
  - task:
      name: LABELED_DEPENDENCIES
      type: token-classification
    metrics:
    - name: Labeled Attachment Score (LAS)
      type: f_score
      value: 0.8140177338
  - task:
      name: SENTS
      type: token-classification
    metrics:
    - name: Sentences F-Score
      type: f_score
      value: 1.0
---
### Details: https://spacy.io/models/ko#ko_core_news_lg

Korean pipeline optimized for CPU. Components: tok2vec, tagger, morphologizer, parser, lemmatizer (trainable_lemmatizer), senter, ner.

| Feature | Description |
| --- | --- |
| **Name** | `ko_core_news_lg` |
| **Version** | `3.5.0` |
| **spaCy** | `>=3.5.0,<3.6.0` |
| **Default Pipeline** | `tok2vec`, `tagger`, `morphologizer`, `parser`, `lemmatizer`, `attribute_ruler`, `ner` |
| **Components** | `tok2vec`, `tagger`, `morphologizer`, `parser`, `lemmatizer`, `senter`, `attribute_ruler`, `ner` |
| **Vectors** | floret (200000, 300) |
| **Sources** | [UD Korean Kaist v2.8](https://github.com/UniversalDependencies/UD_Korean-Kaist) (Choi, Jinho; Han, Na-Rae; Hwang, Jena; Chun, Jayeol)<br />[KLUE v1.1.0](https://github.com/KLUE-benchmark/KLUE) (Sungjoon Park, Jihyung Moon, Sungdong Kim, Won Ik Cho, Jiyoon Han, Jangwon Park, Chisung Song, Junseong Kim, Youngsook Song, Taehwan Oh, Joohong Lee, Juhyun Oh, Sungwon Ryu, Younghoon Jeong, Inkwon Lee, Sangwoo Seo, Dongjun Lee, Hyunwoo Kim, Myeonghwa Lee, Seongbo Jang, Seungwon Do, Sunkyoung Kim, Kyungtae Lim, Jongwon Lee, Kyumin Park, Jamin Shin, Seonghyun Kim, Lucy Park, Alice Oh, Jung-Woo Ha, Kyunghyun Cho)<br />[Explosion Vectors (OSCAR 2109 + Wikipedia + OpenSubtitles + WMT News Crawl)](https://github.com/explosion/spacy-vectors-builder) (Explosion) |
| **License** | `CC BY-SA 4.0` |
| **Author** | [Explosion](https://explosion.ai) |

### Label Scheme

<details>

<summary>View label scheme (2028 labels for 4 components)</summary>

| Component | Labels |
| --- | --- |
| **`tagger`** | `_SP`, `ecs`, `etm`, `f`, `f+f+jcj`, `f+f+jcs`, `f+f+jct`, `f+f+jxt`, `f+jca`, `f+jca+jp+ecc`, `f+jca+jp+ep+ef`, `f+jca+jxc`, `f+jca+jxc+jcm`, `f+jca+jxt`, `f+jcj`, `f+jcm`, `f+jco`, `f+jcs`, `f+jct`, `f+jct+jcm`, `f+jp+ef`, `f+jp+ep+ef`, `f+jp+etm`, `f+jxc`, `f+jxt`, `f+ncn`, `f+ncn+jcm`, `f+ncn+jcs`, `f+ncn+jp+ecc`, `f+ncn+jxt`, `f+ncpa+jcm`, `f+npp+jcs`, `f+nq`, `f+xsn`, `f+xsn+jco`, `f+xsn+jxt`, `ii`, `jca`, `jca+jcm`, `jca+jxc`, `jca+jxt`, `jcc`, `jcj`, `jcm`, `jco`, `jcr`, `jcr+jxc`, `jcs`, `jct`, `jct+jcm`, `jct+jxt`, `jp+ecc`, `jp+ecs`, `jp+ef`, `jp+ef+jcr`, `jp+ef+jcr+jxc`, `jp+ep+ecs`, `jp+ep+ef`, `jp+ep+etm`, `jp+ep+etn`, `jp+etm`, `jp+etn`, `jp+etn+jco`, `jp+etn+jxc`, `jxc`, `jxc+jca`, `jxc+jco`, `jxc+jcs`, `jxt`, `mad`, `mad+jxc`, `mad+jxt`, `mag`, `mag+jca`, `mag+jcm`, `mag+jcs`, `mag+jp+ef+jcr`, `mag+jxc`, `mag+jxc+jxc`, `mag+jxt`, `mag+xsn`, `maj`, `maj+jxc`, `maj+jxt`, `mma`, `mmd`, `nbn`, `nbn+jca`, `nbn+jca+jcj`, `nbn+jca+jcm`, `nbn+jca+jp+ef`, `nbn+jca+jxc`, `nbn+jca+jxt`, `nbn+jcc`, `nbn+jcj`, `nbn+jcm`, `nbn+jco`, `nbn+jcr`, `nbn+jcs`, `nbn+jct`, `nbn+jct+jcm`, `nbn+jct+jxt`, `nbn+jp+ecc`, `nbn+jp+ecs`, `nbn+jp+ecs+jca`, `nbn+jp+ecs+jcm`, `nbn+jp+ecs+jco`, `nbn+jp+ecs+jxc`, `nbn+jp+ecs+jxt`, `nbn+jp+ecx`, `nbn+jp+ef`, `nbn+jp+ef+jca`, `nbn+jp+ef+jco`, `nbn+jp+ef+jcr`, `nbn+jp+ef+jcr+jxc`, `nbn+jp+ef+jcr+jxt`, `nbn+jp+ef+jcs`, `nbn+jp+ef+jxc`, `nbn+jp+ef+jxc+jco`, `nbn+jp+ef+jxf`, `nbn+jp+ef+jxt`, `nbn+jp+ep+ecc`, `nbn+jp+ep+ecs`, `nbn+jp+ep+ecs+jxc`, `nbn+jp+ep+ef`, `nbn+jp+ep+ef+jcr`, `nbn+jp+ep+etm`, `nbn+jp+ep+etn`, `nbn+jp+ep+etn+jco`, `nbn+jp+ep+etn+jcs`, `nbn+jp+etm`, `nbn+jp+etn`, `nbn+jp+etn+jca`, `nbn+jp+etn+jca+jxt`, `nbn+jp+etn+jco`, `nbn+jp+etn+jcs`, `nbn+jp+etn+jxc`, `nbn+jp+etn+jxt`, `nbn+jxc`, `nbn+jxc+jca`, `nbn+jxc+jca+jxc`, `nbn+jxc+jca+jxt`, `nbn+jxc+jcc`, `nbn+jxc+jcm`, `nbn+jxc+jco`, `nbn+jxc+jcs`, `nbn+jxc+jp+ef`, `nbn+jxc+jxc`, `nbn+jxc+jxt`, `nbn+jxt`, `nbn+nbn`, `nbn+nbn+jp+ef`, `nbn+xsm+ecs`, `nbn+xsm+ef`, `nbn+xsm+ep+ef`, `nbn+xsm+ep+ef+jcr`, `nbn+xsm+etm`, `nbn+xsn`, `nbn+xsn+jca`, `nbn+xsn+jca+jp+ef+jcr`, `nbn+xsn+jca+jxc`, `nbn+xsn+jca+jxt`, `nbn+xsn+jcm`, `nbn+xsn+jco`, `nbn+xsn+jcs`, `nbn+xsn+jct`, `nbn+xsn+jp+ecc`, `nbn+xsn+jp+ecs`, `nbn+xsn+jp+ef`, `nbn+xsn+jp+ef+jcr`, `nbn+xsn+jp+ep+ef`, `nbn+xsn+jxc`, `nbn+xsn+jxt`, `nbn+xsv+etm`, `nbu`, `nbu+jca`, `nbu+jca+jxc`, `nbu+jca+jxt`, `nbu+jcc`, `nbu+jcc+jxc`, `nbu+jcj`, `nbu+jcm`, `nbu+jco`, `nbu+jcs`, `nbu+jct`, `nbu+jct+jxc`, `nbu+jp+ecc`, `nbu+jp+ecs`, `nbu+jp+ef`, `nbu+jp+ef+jcr`, `nbu+jp+ef+jxc`, `nbu+jp+ep+ecc`, `nbu+jp+ep+ecs`, `nbu+jp+ep+ef`, `nbu+jp+ep+ef+jcr`, `nbu+jp+ep+etm`, `nbu+jp+ep+etn+jco`, `nbu+jp+etm`, `nbu+jxc`, `nbu+jxc+jca`, `nbu+jxc+jcs`, `nbu+jxc+jp+ef`, `nbu+jxc+jp+ep+ef`, `nbu+jxc+jxt`, `nbu+jxt`, `nbu+ncn`, `nbu+ncn+jca`, `nbu+ncn+jcm`, `nbu+xsn`, `nbu+xsn+jca`, `nbu+xsn+jca+jxc`, `nbu+xsn+jca+jxt`, `nbu+xsn+jcm`, `nbu+xsn+jco`, `nbu+xsn+jcs`, `nbu+xsn+jp+ecs`, `nbu+xsn+jp+ep+ef`, `nbu+xsn+jxc`, `nbu+xsn+jxc+jxt`, `nbu+xsn+jxt`, `nbu+xsv+ecc`, `nbu+xsv+etm`, `ncn`, `ncn+f+ncpa+jco`, `ncn+jca`, `ncn+jca+jca`, `ncn+jca+jcc`, `ncn+jca+jcj`, `ncn+jca+jcm`, `ncn+jca+jcs`, `ncn+jca+jct`, `ncn+jca+jp+ecc`, `ncn+jca+jp+ecs`, `ncn+jca+jp+ef`, `ncn+jca+jp+ep+ef`, `ncn+jca+jp+etm`, `ncn+jca+jp+etn+jxt`, `ncn+jca+jxc`, `ncn+jca+jxc+jcc`, `ncn+jca+jxc+jcm`, `ncn+jca+jxc+jxc`, `ncn+jca+jxc+jxt`, `ncn+jca+jxt`, `ncn+jcc`, `ncn+jcc+jxc`, `ncn+jcj`, `ncn+jcj+jxt`, `ncn+jcm`, `ncn+jco`, `ncn+jcr`, `ncn+jcr+jxc`, `ncn+jcs`, `ncn+jcs+jxt`, `ncn+jct`, `ncn+jct+jcm`, `ncn+jct+jxc`, `ncn+jct+jxt`, `ncn+jcv`, `ncn+jp+ecc`, `ncn+jp+ecc+jct`, `ncn+jp+ecc+jxc`, `ncn+jp+ecs`, `ncn+jp+ecs+jcm`, `ncn+jp+ecs+jco`, `ncn+jp+ecs+jxc`, `ncn+jp+ecs+jxt`, `ncn+jp+ecx`, `ncn+jp+ef`, `ncn+jp+ef+jca`, `ncn+jp+ef+jcm`, `ncn+jp+ef+jco`, `ncn+jp+ef+jcr`, `ncn+jp+ef+jcr+jxc`, `ncn+jp+ef+jcr+jxt`, `ncn+jp+ef+jp+etm`, `ncn+jp+ef+jxc`, `ncn+jp+ef+jxf`, `ncn+jp+ef+jxt`, `ncn+jp+ep+ecc`, `ncn+jp+ep+ecs`, `ncn+jp+ep+ecs+jxc`, `ncn+jp+ep+ecx`, `ncn+jp+ep+ef`, `ncn+jp+ep+ef+jcr`, `ncn+jp+ep+ef+jcr+jxc`, `ncn+jp+ep+ef+jxc`, `ncn+jp+ep+ef+jxf`, `ncn+jp+ep+ef+jxt`, `ncn+jp+ep+ep+etm`, `ncn+jp+ep+etm`, `ncn+jp+ep+etn`, `ncn+jp+ep+etn+jca`, `ncn+jp+ep+etn+jca+jxc`, `ncn+jp+ep+etn+jco`, `ncn+jp+ep+etn+jcs`, `ncn+jp+ep+etn+jxt`, `ncn+jp+etm`, `ncn+jp+etn`, `ncn+jp+etn+jca`, `ncn+jp+etn+jca+jxc`, `ncn+jp+etn+jca+jxt`, `ncn+jp+etn+jco`, `ncn+jp+etn+jcs`, `ncn+jp+etn+jct`, `ncn+jp+etn+jxc`, `ncn+jp+etn+jxt`, `ncn+jxc`, `ncn+jxc+jca`, `ncn+jxc+jca+jxc`, `ncn+jxc+jca+jxt`, `ncn+jxc+jcc`, `ncn+jxc+jcm`, `ncn+jxc+jco`, `ncn+jxc+jcs`, `ncn+jxc+jct+jxt`, `ncn+jxc+jp+ef`, `ncn+jxc+jp+ef+jcr`, `ncn+jxc+jp+ep+ecs`, `ncn+jxc+jp+ep+ef`, `ncn+jxc+jp+etm`, `ncn+jxc+jxc`, `ncn+jxc+jxt`, `ncn+jxt`, `ncn+jxt+jcm`, `ncn+jxt+jxc`, `ncn+nbn`, `ncn+nbn+jca`, `ncn+nbn+jcm`, `ncn+nbn+jcs`, `ncn+nbn+jp+ecc`, `ncn+nbn+jp+ep+ef`, `ncn+nbn+jxc`, `ncn+nbn+jxt`, `ncn+nbu`, `ncn+nbu+jca`, `ncn+nbu+jcm`, `ncn+nbu+jco`, `ncn+nbu+jp+ef`, `ncn+nbu+jxc`, `ncn+nbu+ncn`, `ncn+ncn`, `ncn+ncn+jca`, `ncn+ncn+jca+jcc`, `ncn+ncn+jca+jcm`, `ncn+ncn+jca+jxc`, `ncn+ncn+jca+jxc+jcm`, `ncn+ncn+jca+jxc+jxc`, `ncn+ncn+jca+jxt`, `ncn+ncn+jcc`, `ncn+ncn+jcj`, `ncn+ncn+jcm`, `ncn+ncn+jco`, `ncn+ncn+jcr`, `ncn+ncn+jcs`, `ncn+ncn+jct`, `ncn+ncn+jct+jcm`, `ncn+ncn+jct+jxc`, `ncn+ncn+jct+jxt`, `ncn+ncn+jp+ecc`, `ncn+ncn+jp+ecs`, `ncn+ncn+jp+ef`, `ncn+ncn+jp+ef+jcm`, `ncn+ncn+jp+ef+jcr`, `ncn+ncn+jp+ef+jcs`, `ncn+ncn+jp+ep+ecc`, `ncn+ncn+jp+ep+ecs`, `ncn+ncn+jp+ep+ef`, `ncn+ncn+jp+ep+ef+jcr`, `ncn+ncn+jp+ep+ep+etm`, `ncn+ncn+jp+ep+etm`, `ncn+ncn+jp+ep+etn`, `ncn+ncn+jp+etm`, `ncn+ncn+jp+etn`, `ncn+ncn+jp+etn+jca`, `ncn+ncn+jp+etn+jco`, `ncn+ncn+jp+etn+jxc`, `ncn+ncn+jxc`, `ncn+ncn+jxc+jca`, `ncn+ncn+jxc+jcc`, `ncn+ncn+jxc+jcm`, `ncn+ncn+jxc+jco`, `ncn+ncn+jxc+jcs`, `ncn+ncn+jxc+jxc`, `ncn+ncn+jxt`, `ncn+ncn+nbn`, `ncn+ncn+ncn`, `ncn+ncn+ncn+jca`, `ncn+ncn+ncn+jca+jcm`, `ncn+ncn+ncn+jca+jxt`, `ncn+ncn+ncn+jcj`, `ncn+ncn+ncn+jcm`, `ncn+ncn+ncn+jco`, `ncn+ncn+ncn+jcs`, `ncn+ncn+ncn+jct+jxt`, `ncn+ncn+ncn+jp+etn+jxc`, `ncn+ncn+ncn+jxt`, `ncn+ncn+ncn+ncn+jca`, `ncn+ncn+ncn+ncn+jca+jxt`, `ncn+ncn+ncn+ncn+jco`, `ncn+ncn+ncn+xsn+jp+etm`, `ncn+ncn+ncpa`, `ncn+ncn+ncpa+jca`, `ncn+ncn+ncpa+jcm`, `ncn+ncn+ncpa+jco`, `ncn+ncn+ncpa+jcs`, `ncn+ncn+ncpa+jxc`, `ncn+ncn+ncpa+jxt`, `ncn+ncn+ncpa+ncn`, `ncn+ncn+ncpa+ncn+jca`, `ncn+ncn+ncpa+ncn+jcj`, `ncn+ncn+ncpa+ncn+jcm`, `ncn+ncn+ncpa+ncn+jxt`, `ncn+ncn+xsn`, `ncn+ncn+xsn+jca`, `ncn+ncn+xsn+jca+jxt`, `ncn+ncn+xsn+jcj`, `ncn+ncn+xsn+jcm`, `ncn+ncn+xsn+jco`, `ncn+ncn+xsn+jcs`, `ncn+ncn+xsn+jct`, `ncn+ncn+xsn+jp+ecs`, `ncn+ncn+xsn+jp+ep+ef`, `ncn+ncn+xsn+jp+etm`, `ncn+ncn+xsn+jxc`, `ncn+ncn+xsn+jxc+jcs`, `ncn+ncn+xsn+jxt`, `ncn+ncn+xsv+ecc`, `ncn+ncn+xsv+etm`, `ncn+ncpa`, `ncn+ncpa+jca`, `ncn+ncpa+jca+jcm`, `ncn+ncpa+jca+jxc`, `ncn+ncpa+jca+jxt`, `ncn+ncpa+jcc`, `ncn+ncpa+jcj`, `ncn+ncpa+jcm`, `ncn+ncpa+jco`, `ncn+ncpa+jcr`, `ncn+ncpa+jcs`, `ncn+ncpa+jct`, `ncn+ncpa+jct+jcm`, `ncn+ncpa+jct+jxt`, `ncn+ncpa+jp+ecc`, `ncn+ncpa+jp+ecc+jxc`, `ncn+ncpa+jp+ecs`, `ncn+ncpa+jp+ecs+jxc`, `ncn+ncpa+jp+ef`, `ncn+ncpa+jp+ef+jcr`, `ncn+ncpa+jp+ef+jcr+jxc`, `ncn+ncpa+jp+ep+ef`, `ncn+ncpa+jp+ep+etm`, `ncn+ncpa+jp+ep+etn`, `ncn+ncpa+jp+etm`, `ncn+ncpa+jxc`, `ncn+ncpa+jxc+jca+jxc`, `ncn+ncpa+jxc+jco`, `ncn+ncpa+jxc+jcs`, `ncn+ncpa+jxt`, `ncn+ncpa+nbn+jcs`, `ncn+ncpa+ncn`, `ncn+ncpa+ncn+jca`, `ncn+ncpa+ncn+jca+jcm`, `ncn+ncpa+ncn+jca+jxc`, `ncn+ncpa+ncn+jca+jxt`, `ncn+ncpa+ncn+jcj`, `ncn+ncpa+ncn+jcm`, `ncn+ncpa+ncn+jco`, `ncn+ncpa+ncn+jcs`, `ncn+ncpa+ncn+jct`, `ncn+ncpa+ncn+jct+jcm`, `ncn+ncpa+ncn+jp+ef+jcr`, `ncn+ncpa+ncn+jp+ep+etm`, `ncn+ncpa+ncn+jxc`, `ncn+ncpa+ncn+jxt`, `ncn+ncpa+ncn+xsn+jcm`, `ncn+ncpa+ncn+xsn+jxt`, `ncn+ncpa+ncpa`, `ncn+ncpa+ncpa+jca`, `ncn+ncpa+ncpa+jcj`, `ncn+ncpa+ncpa+jcm`, `ncn+ncpa+ncpa+jco`, `ncn+ncpa+ncpa+jcs`, `ncn+ncpa+ncpa+jp+ep+ef`, `ncn+ncpa+ncpa+jxt`, `ncn+ncpa+ncpa+ncn`, `ncn+ncpa+xsn`, `ncn+ncpa+xsn+jcm`, `ncn+ncpa+xsn+jco`, `ncn+ncpa+xsn+jcs`, `ncn+ncpa+xsn+jp+ecc`, `ncn+ncpa+xsn+jp+etm`, `ncn+ncpa+xsn+jxt`, `ncn+ncpa+xsv+ecc`, `ncn+ncpa+xsv+ecs`, `ncn+ncpa+xsv+ecx`, `ncn+ncpa+xsv+ecx+px+etm`, `ncn+ncpa+xsv+ef`, `ncn+ncpa+xsv+ef+jcm`, `ncn+ncpa+xsv+ef+jcr`, `ncn+ncpa+xsv+etm`, _(truncated: full list in pipeline meta)_ |
| **`morphologizer`** | `POS=CCONJ`, `POS=ADV`, `POS=SCONJ`, `POS=DET`, `POS=NOUN`, `POS=VERB`, `POS=ADJ`, `POS=PUNCT`, `POS=SPACE`, `POS=AUX`, `POS=PRON`, `POS=PROPN`, `POS=NUM`, `POS=INTJ`, `POS=PART`, `POS=X`, `POS=ADP`, `POS=SYM` |
| **`parser`** | `ROOT`, `acl`, `advcl`, `advmod`, `amod`, `appos`, `aux`, `case`, `cc`, `ccomp`, `compound`, `conj`, `cop`, `csubj`, `dep`, `det`, `dislocated`, `fixed`, `flat`, `iobj`, `mark`, `nmod`, `nsubj`, `nummod`, `obj`, `obl`, `punct`, `xcomp` |
| **`ner`** | `DT`, `LC`, `OG`, `PS`, `QT`, `TI` |

</details>

### Accuracy

| Type | Score |
| --- | --- |
| `TOKEN_ACC` | 100.00 |
| `TOKEN_P` | 100.00 |
| `TOKEN_R` | 100.00 |
| `TOKEN_F` | 100.00 |
| `TAG_ACC` | 84.00 |
| `POS_ACC` | 94.88 |
| `SENTS_P` | 100.00 |
| `SENTS_R` | 100.00 |
| `SENTS_F` | 100.00 |
| `DEP_UAS` | 84.17 |
| `DEP_LAS` | 81.40 |
| `LEMMA_ACC` | 90.09 |
| `ENTS_P` | 86.69 |
| `ENTS_R` | 83.73 |
| `ENTS_F` | 85.19 |
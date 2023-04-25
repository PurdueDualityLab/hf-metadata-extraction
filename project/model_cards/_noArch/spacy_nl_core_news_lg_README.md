---
tags:
- spacy
- token-classification
language:
- nl
license: cc-by-sa-4.0
model-index:
- name: nl_core_news_lg
  results:
  - task:
      name: NER
      type: token-classification
    metrics:
    - name: NER Precision
      type: precision
      value: 0.7850940666
    - name: NER Recall
      type: recall
      value: 0.7503457815
    - name: NER F Score
      type: f_score
      value: 0.7673267327
  - task:
      name: TAG
      type: token-classification
    metrics:
    - name: TAG (XPOS) Accuracy
      type: accuracy
      value: 0.9514067612
  - task:
      name: POS
      type: token-classification
    metrics:
    - name: POS (UPOS) Accuracy
      type: accuracy
      value: 0.9638822246
  - task:
      name: MORPH
      type: token-classification
    metrics:
    - name: Morph (UFeats) Accuracy
      type: accuracy
      value: 0.9628967172
  - task:
      name: LEMMA
      type: token-classification
    metrics:
    - name: Lemma Accuracy
      type: accuracy
      value: 0.9556229147
  - task:
      name: UNLABELED_DEPENDENCIES
      type: token-classification
    metrics:
    - name: Unlabeled Attachment Score (UAS)
      type: f_score
      value: 0.8702417761
  - task:
      name: LABELED_DEPENDENCIES
      type: token-classification
    metrics:
    - name: Labeled Attachment Score (LAS)
      type: f_score
      value: 0.8253421186
  - task:
      name: SENTS
      type: token-classification
    metrics:
    - name: Sentences F-Score
      type: f_score
      value: 0.8731501057
---
### Details: https://spacy.io/models/nl#nl_core_news_lg

Dutch pipeline optimized for CPU. Components: tok2vec, morphologizer, tagger, parser, lemmatizer (trainable_lemmatizer), senter, ner.

| Feature | Description |
| --- | --- |
| **Name** | `nl_core_news_lg` |
| **Version** | `3.5.0` |
| **spaCy** | `>=3.5.0,<3.6.0` |
| **Default Pipeline** | `tok2vec`, `morphologizer`, `tagger`, `parser`, `lemmatizer`, `attribute_ruler`, `ner` |
| **Components** | `tok2vec`, `morphologizer`, `tagger`, `parser`, `lemmatizer`, `senter`, `attribute_ruler`, `ner` |
| **Vectors** | 500000 keys, 500000 unique vectors (300 dimensions) |
| **Sources** | [UD Dutch LassySmall v2.8](https://github.com/UniversalDependencies/UD_Dutch-LassySmall) (Bouma, Gosse; van Noord, Gertjan)<br />[Dutch NER Annotations for UD LassySmall](https://nlp.town) (NLP Town)<br />[UD Dutch Alpino v2.8](https://github.com/UniversalDependencies/UD_Dutch-Alpino) (Zeman, Daniel; Žabokrtský, Zdeněk; Bouma, Gosse; van Noord, Gertjan)<br />[Explosion fastText Vectors (cbow, OSCAR Common Crawl + Wikipedia)](https://spacy.io) (Explosion) |
| **License** | `CC BY-SA 4.0` |
| **Author** | [Explosion](https://explosion.ai) |

### Label Scheme

<details>

<summary>View label scheme (323 labels for 4 components)</summary>

| Component | Labels |
| --- | --- |
| **`morphologizer`** | `POS=PRON\|Person=3\|PronType=Dem`, `Number=Sing\|POS=AUX\|Tense=Pres\|VerbForm=Fin`, `POS=ADV`, `POS=VERB\|VerbForm=Part`, `POS=PUNCT`, `Number=Sing\|POS=AUX\|Tense=Past\|VerbForm=Fin`, `POS=ADP`, `POS=NUM`, `Number=Plur\|POS=NOUN`, `POS=VERB\|VerbForm=Inf`, `POS=SCONJ`, `Definite=Def\|POS=DET`, `Gender=Com\|Number=Sing\|POS=NOUN`, `Number=Sing\|POS=VERB\|Tense=Pres\|VerbForm=Fin`, `Degree=Pos\|POS=ADJ`, `Gender=Neut\|Number=Sing\|POS=PROPN`, `Gender=Com\|Number=Sing\|POS=PROPN`, `POS=AUX\|VerbForm=Inf`, `Number=Sing\|POS=VERB\|Tense=Past\|VerbForm=Fin`, `POS=DET`, `Gender=Neut\|Number=Sing\|POS=NOUN`, `POS=PRON\|Person=3\|PronType=Prs`, `POS=CCONJ`, `Number=Plur\|POS=VERB\|Tense=Pres\|VerbForm=Fin`, `POS=PRON\|Person=3\|PronType=Ind`, `Degree=Cmp\|POS=ADJ`, `Case=Nom\|POS=PRON\|Person=1\|PronType=Prs`, `Definite=Ind\|POS=DET`, `Case=Nom\|POS=PRON\|Person=3\|PronType=Prs`, `POS=PRON\|Person=3\|Poss=Yes\|PronType=Prs`, `Number=Plur\|POS=AUX\|Tense=Pres\|VerbForm=Fin`, `POS=PRON\|PronType=Rel`, `Case=Acc\|POS=PRON\|Person=1\|PronType=Prs`, `Number=Plur\|POS=VERB\|Tense=Past\|VerbForm=Fin`, `Gender=Com,Neut\|Number=Sing\|POS=NOUN`, `Case=Acc\|POS=PRON\|Person=3\|PronType=Prs\|Reflex=Yes`, `Case=Acc\|POS=PRON\|Person=3\|PronType=Prs`, `POS=PROPN`, `POS=PRON\|PronType=Ind`, `POS=PRON\|Person=3\|PronType=Int`, `Case=Acc\|POS=PRON\|PronType=Rcp`, `Number=Plur\|POS=AUX\|Tense=Past\|VerbForm=Fin`, `Number=Sing\|POS=NOUN`, `POS=PRON\|Person=1\|Poss=Yes\|PronType=Prs`, `POS=SYM`, `Abbr=Yes\|POS=X`, `Gender=Com,Neut\|Number=Sing\|POS=PROPN`, `Degree=Sup\|POS=ADJ`, `POS=ADJ`, `Number=Sing\|POS=PROPN`, `POS=PRON\|PronType=Dem`, `POS=AUX\|VerbForm=Part`, `POS=SPACE`, `POS=PRON\|Person=3\|PronType=Rel`, `Number=Plur\|POS=PROPN`, `POS=PRON\|Person=2\|Poss=Yes\|PronType=Prs`, `Case=Dat\|POS=PRON\|PronType=Dem`, `Case=Nom\|POS=PRON\|Person=2\|PronType=Prs`, `POS=INTJ`, `Case=Acc\|POS=PRON\|Person=2\|PronType=Prs`, `Case=Gen\|POS=PRON\|Person=3\|Poss=Yes\|PronType=Prs`, `POS=PRON\|PronType=Int`, `POS=PRON\|Person=2\|PronType=Prs`, `POS=PRON\|Person=3`, `Case=Gen\|POS=PRON\|Person=2\|PronType=Prs`, `POS=X` |
| **`tagger`** | `ADJ\|nom\|basis\|met-e\|mv-n`, `ADJ\|nom\|basis\|met-e\|zonder-n\|bijz`, `ADJ\|nom\|basis\|met-e\|zonder-n\|stan`, `ADJ\|nom\|basis\|zonder\|mv-n`, `ADJ\|nom\|basis\|zonder\|zonder-n`, `ADJ\|nom\|comp\|met-e\|mv-n`, `ADJ\|nom\|comp\|met-e\|zonder-n\|stan`, `ADJ\|nom\|sup\|met-e\|mv-n`, `ADJ\|nom\|sup\|met-e\|zonder-n\|bijz`, `ADJ\|nom\|sup\|met-e\|zonder-n\|stan`, `ADJ\|nom\|sup\|zonder\|zonder-n`, `ADJ\|postnom\|basis\|met-s`, `ADJ\|postnom\|basis\|zonder`, `ADJ\|postnom\|comp\|met-s`, `ADJ\|prenom\|basis\|met-e\|bijz`, `ADJ\|prenom\|basis\|met-e\|stan`, `ADJ\|prenom\|basis\|zonder`, `ADJ\|prenom\|comp\|met-e\|stan`, `ADJ\|prenom\|comp\|zonder`, `ADJ\|prenom\|sup\|met-e\|stan`, `ADJ\|prenom\|sup\|zonder`, `ADJ\|vrij\|basis\|zonder`, `ADJ\|vrij\|comp\|zonder`, `ADJ\|vrij\|dim\|zonder`, `ADJ\|vrij\|sup\|zonder`, `BW`, `LET`, `LID\|bep\|dat\|evmo`, `LID\|bep\|gen\|evmo`, `LID\|bep\|gen\|rest3`, `LID\|bep\|stan\|evon`, `LID\|bep\|stan\|rest`, `LID\|onbep\|stan\|agr`, `N\|eigen\|ev\|basis\|gen`, `N\|eigen\|ev\|basis\|genus\|stan`, `N\|eigen\|ev\|basis\|onz\|stan`, `N\|eigen\|ev\|basis\|zijd\|stan`, `N\|eigen\|ev\|dim\|onz\|stan`, `N\|eigen\|mv\|basis`, `N\|soort\|ev\|basis\|dat`, `N\|soort\|ev\|basis\|gen`, `N\|soort\|ev\|basis\|genus\|stan`, `N\|soort\|ev\|basis\|onz\|stan`, `N\|soort\|ev\|basis\|zijd\|stan`, `N\|soort\|ev\|dim\|onz\|stan`, `N\|soort\|mv\|basis`, `N\|soort\|mv\|dim`, `SPEC\|afgebr`, `SPEC\|afk`, `SPEC\|deeleigen`, `SPEC\|enof`, `SPEC\|meta`, `SPEC\|symb`, `SPEC\|vreemd`, `TSW`, `TW\|hoofd\|nom\|mv-n\|basis`, `TW\|hoofd\|nom\|mv-n\|dim`, `TW\|hoofd\|nom\|zonder-n\|basis`, `TW\|hoofd\|nom\|zonder-n\|dim`, `TW\|hoofd\|prenom\|stan`, `TW\|hoofd\|vrij`, `TW\|rang\|nom\|mv-n`, `TW\|rang\|nom\|zonder-n`, `TW\|rang\|prenom\|stan`, `VG\|neven`, `VG\|onder`, `VNW\|aanw\|adv-pron\|obl\|vol\|3o\|getal`, `VNW\|aanw\|adv-pron\|stan\|red\|3\|getal`, `VNW\|aanw\|det\|dat\|nom\|met-e\|zonder-n`, `VNW\|aanw\|det\|dat\|prenom\|met-e\|evmo`, `VNW\|aanw\|det\|gen\|prenom\|met-e\|rest3`, `VNW\|aanw\|det\|stan\|nom\|met-e\|mv-n`, `VNW\|aanw\|det\|stan\|nom\|met-e\|zonder-n`, `VNW\|aanw\|det\|stan\|prenom\|met-e\|rest`, `VNW\|aanw\|det\|stan\|prenom\|zonder\|agr`, `VNW\|aanw\|det\|stan\|prenom\|zonder\|evon`, `VNW\|aanw\|det\|stan\|prenom\|zonder\|rest`, `VNW\|aanw\|det\|stan\|vrij\|zonder`, `VNW\|aanw\|pron\|gen\|vol\|3m\|ev`, `VNW\|aanw\|pron\|stan\|vol\|3o\|ev`, `VNW\|aanw\|pron\|stan\|vol\|3\|getal`, `VNW\|betr\|det\|stan\|nom\|met-e\|zonder-n`, `VNW\|betr\|det\|stan\|nom\|zonder\|zonder-n`, `VNW\|betr\|pron\|stan\|vol\|3\|ev`, `VNW\|betr\|pron\|stan\|vol\|persoon\|getal`, `VNW\|bez\|det\|gen\|vol\|3\|ev\|prenom\|met-e\|rest3`, `VNW\|bez\|det\|stan\|nadr\|2v\|mv\|prenom\|zonder\|agr`, `VNW\|bez\|det\|stan\|red\|1\|ev\|prenom\|zonder\|agr`, `VNW\|bez\|det\|stan\|red\|2v\|ev\|prenom\|zonder\|agr`, `VNW\|bez\|det\|stan\|red\|3\|ev\|prenom\|zonder\|agr`, `VNW\|bez\|det\|stan\|vol\|1\|ev\|prenom\|met-e\|rest`, `VNW\|bez\|det\|stan\|vol\|1\|ev\|prenom\|zonder\|agr`, `VNW\|bez\|det\|stan\|vol\|1\|mv\|prenom\|met-e\|rest`, `VNW\|bez\|det\|stan\|vol\|1\|mv\|prenom\|zonder\|evon`, `VNW\|bez\|det\|stan\|vol\|2v\|ev\|prenom\|zonder\|agr`, `VNW\|bez\|det\|stan\|vol\|2\|getal\|prenom\|zonder\|agr`, `VNW\|bez\|det\|stan\|vol\|3m\|ev\|nom\|met-e\|zonder-n`, `VNW\|bez\|det\|stan\|vol\|3m\|ev\|prenom\|met-e\|rest`, `VNW\|bez\|det\|stan\|vol\|3p\|mv\|prenom\|met-e\|rest`, `VNW\|bez\|det\|stan\|vol\|3v\|ev\|nom\|met-e\|zonder-n`, `VNW\|bez\|det\|stan\|vol\|3v\|ev\|prenom\|met-e\|rest`, `VNW\|bez\|det\|stan\|vol\|3\|ev\|prenom\|zonder\|agr`, `VNW\|bez\|det\|stan\|vol\|3\|mv\|prenom\|zonder\|agr`, `VNW\|excl\|pron\|stan\|vol\|3\|getal`, `VNW\|onbep\|adv-pron\|gen\|red\|3\|getal`, `VNW\|onbep\|adv-pron\|obl\|vol\|3o\|getal`, `VNW\|onbep\|det\|stan\|nom\|met-e\|mv-n`, `VNW\|onbep\|det\|stan\|nom\|met-e\|zonder-n`, `VNW\|onbep\|det\|stan\|nom\|zonder\|zonder-n`, `VNW\|onbep\|det\|stan\|prenom\|met-e\|agr`, `VNW\|onbep\|det\|stan\|prenom\|met-e\|evz`, `VNW\|onbep\|det\|stan\|prenom\|met-e\|mv`, `VNW\|onbep\|det\|stan\|prenom\|met-e\|rest`, `VNW\|onbep\|det\|stan\|prenom\|zonder\|agr`, `VNW\|onbep\|det\|stan\|prenom\|zonder\|evon`, `VNW\|onbep\|det\|stan\|vrij\|zonder`, `VNW\|onbep\|grad\|gen\|nom\|met-e\|mv-n\|basis`, `VNW\|onbep\|grad\|stan\|nom\|met-e\|mv-n\|basis`, `VNW\|onbep\|grad\|stan\|nom\|met-e\|mv-n\|sup`, `VNW\|onbep\|grad\|stan\|nom\|met-e\|zonder-n\|basis`, `VNW\|onbep\|grad\|stan\|nom\|met-e\|zonder-n\|sup`, `VNW\|onbep\|grad\|stan\|prenom\|met-e\|agr\|basis`, `VNW\|onbep\|grad\|stan\|prenom\|met-e\|agr\|comp`, `VNW\|onbep\|grad\|stan\|prenom\|met-e\|agr\|sup`, `VNW\|onbep\|grad\|stan\|prenom\|met-e\|mv\|basis`, `VNW\|onbep\|grad\|stan\|prenom\|zonder\|agr\|basis`, `VNW\|onbep\|grad\|stan\|prenom\|zonder\|agr\|comp`, `VNW\|onbep\|grad\|stan\|vrij\|zonder\|basis`, `VNW\|onbep\|grad\|stan\|vrij\|zonder\|comp`, `VNW\|onbep\|grad\|stan\|vrij\|zonder\|sup`, `VNW\|onbep\|pron\|gen\|vol\|3p\|ev`, `VNW\|onbep\|pron\|stan\|vol\|3o\|ev`, `VNW\|onbep\|pron\|stan\|vol\|3p\|ev`, `VNW\|pers\|pron\|gen\|vol\|2\|getal`, `VNW\|pers\|pron\|nomin\|nadr\|3m\|ev\|masc`, `VNW\|pers\|pron\|nomin\|nadr\|3v\|ev\|fem`, `VNW\|pers\|pron\|nomin\|red\|1\|mv`, `VNW\|pers\|pron\|nomin\|red\|2v\|ev`, `VNW\|pers\|pron\|nomin\|red\|2\|getal`, `VNW\|pers\|pron\|nomin\|red\|3p\|ev\|masc`, `VNW\|pers\|pron\|nomin\|red\|3\|ev\|masc`, `VNW\|pers\|pron\|nomin\|vol\|1\|ev`, `VNW\|pers\|pron\|nomin\|vol\|1\|mv`, `VNW\|pers\|pron\|nomin\|vol\|2b\|getal`, `VNW\|pers\|pron\|nomin\|vol\|2v\|ev`, `VNW\|pers\|pron\|nomin\|vol\|2\|getal`, `VNW\|pers\|pron\|nomin\|vol\|3p\|mv`, `VNW\|pers\|pron\|nomin\|vol\|3v\|ev\|fem`, `VNW\|pers\|pron\|nomin\|vol\|3\|ev\|masc`, `VNW\|pers\|pron\|obl\|nadr\|3m\|ev\|masc`, `VNW\|pers\|pron\|obl\|red\|3\|ev\|masc`, `VNW\|pers\|pron\|obl\|vol\|2v\|ev`, `VNW\|pers\|pron\|obl\|vol\|3p\|mv`, `VNW\|pers\|pron\|obl\|vol\|3\|ev\|masc`, `VNW\|pers\|pron\|obl\|vol\|3\|getal\|fem`, `VNW\|pers\|pron\|stan\|nadr\|2v\|mv`, `VNW\|pers\|pron\|stan\|red\|3\|ev\|fem`, `VNW\|pers\|pron\|stan\|red\|3\|ev\|onz`, `VNW\|pers\|pron\|stan\|red\|3\|mv`, `VNW\|pr\|pron\|obl\|nadr\|1\|ev`, `VNW\|pr\|pron\|obl\|nadr\|2v\|getal`, `VNW\|pr\|pron\|obl\|nadr\|2\|getal`, `VNW\|pr\|pron\|obl\|red\|1\|ev`, `VNW\|pr\|pron\|obl\|red\|2v\|getal`, `VNW\|pr\|pron\|obl\|vol\|1\|ev`, `VNW\|pr\|pron\|obl\|vol\|1\|mv`, `VNW\|pr\|pron\|obl\|vol\|2\|getal`, `VNW\|recip\|pron\|gen\|vol\|persoon\|mv`, `VNW\|recip\|pron\|obl\|vol\|persoon\|mv`, `VNW\|refl\|pron\|obl\|nadr\|3\|getal`, `VNW\|refl\|pron\|obl\|red\|3\|getal`, `VNW\|vb\|adv-pron\|obl\|vol\|3o\|getal`, `VNW\|vb\|det\|stan\|nom\|met-e\|zonder-n`, `VNW\|vb\|det\|stan\|prenom\|met-e\|rest`, `VNW\|vb\|det\|stan\|prenom\|zonder\|evon`, `VNW\|vb\|pron\|gen\|vol\|3m\|ev`, `VNW\|vb\|pron\|gen\|vol\|3p\|mv`, `VNW\|vb\|pron\|gen\|vol\|3v\|ev`, `VNW\|vb\|pron\|stan\|vol\|3o\|ev`, `VNW\|vb\|pron\|stan\|vol\|3p\|getal`, `VZ\|fin`, `VZ\|init`, `VZ\|versm`, `WW\|inf\|nom\|zonder\|zonder-n`, `WW\|inf\|prenom\|met-e`, `WW\|inf\|vrij\|zonder`, `WW\|od\|nom\|met-e\|mv-n`, `WW\|od\|nom\|met-e\|zonder-n`, `WW\|od\|prenom\|met-e`, `WW\|od\|prenom\|zonder`, `WW\|od\|vrij\|zonder`, `WW\|pv\|conj\|ev`, `WW\|pv\|tgw\|ev`, `WW\|pv\|tgw\|met-t`, `WW\|pv\|tgw\|mv`, `WW\|pv\|verl\|ev`, `WW\|pv\|verl\|mv`, `WW\|vd\|nom\|met-e\|mv-n`, `WW\|vd\|nom\|met-e\|zonder-n`, `WW\|vd\|prenom\|met-e`, `WW\|vd\|prenom\|zonder`, `WW\|vd\|vrij\|zonder`, `_SP` |
| **`parser`** | `ROOT`, `acl`, `acl:relcl`, `advcl`, `advmod`, `amod`, `appos`, `aux`, `aux:pass`, `case`, `cc`, `ccomp`, `compound:prt`, `conj`, `cop`, `csubj`, `dep`, `det`, `expl`, `expl:pv`, `fixed`, `flat`, `iobj`, `mark`, `nmod`, `nmod:poss`, `nsubj`, `nsubj:pass`, `nummod`, `obj`, `obl`, `obl:agent`, `orphan`, `parataxis`, `punct`, `xcomp` |
| **`ner`** | `CARDINAL`, `DATE`, `EVENT`, `FAC`, `GPE`, `LANGUAGE`, `LAW`, `LOC`, `MONEY`, `NORP`, `ORDINAL`, `ORG`, `PERCENT`, `PERSON`, `PRODUCT`, `QUANTITY`, `TIME`, `WORK_OF_ART` |

</details>

### Accuracy

| Type | Score |
| --- | --- |
| `TAG_ACC` | 95.14 |
| `SENTS_P` | 85.80 |
| `SENTS_R` | 88.88 |
| `SENTS_F` | 87.32 |
| `DEP_UAS` | 87.02 |
| `DEP_LAS` | 82.53 |
| `ENTS_P` | 78.51 |
| `ENTS_R` | 75.03 |
| `ENTS_F` | 76.73 |
| `TOKEN_ACC` | 99.94 |
| `TOKEN_P` | 99.74 |
| `TOKEN_R` | 99.76 |
| `TOKEN_F` | 99.75 |
| `POS_ACC` | 96.39 |
| `MORPH_ACC` | 96.29 |
| `MORPH_MICRO_P` | 97.12 |
| `MORPH_MICRO_R` | 95.47 |
| `MORPH_MICRO_F` | 96.29 |
| `LEMMA_ACC` | 95.56 |
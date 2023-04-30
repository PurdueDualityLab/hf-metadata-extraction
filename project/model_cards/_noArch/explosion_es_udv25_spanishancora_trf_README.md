---
tags:
- spacy
- token-classification
language:
- es
license: gpl-3.0
model-index:
- name: es_udv25_spanishancora_trf
  results:
  - task:
      name: TAG
      type: token-classification
    metrics:
    - name: TAG (XPOS) Accuracy
      type: accuracy
      value: 0.9891586707
  - task:
      name: POS
      type: token-classification
    metrics:
    - name: POS (UPOS) Accuracy
      type: accuracy
      value: 0.9903472868
  - task:
      name: MORPH
      type: token-classification
    metrics:
    - name: Morph (UFeats) Accuracy
      type: accuracy
      value: 0.9795632752
  - task:
      name: LEMMA
      type: token-classification
    metrics:
    - name: Lemma Accuracy
      type: accuracy
      value: 0.9892930745
  - task:
      name: UNLABELED_DEPENDENCIES
      type: token-classification
    metrics:
    - name: Unlabeled Attachment Score (UAS)
      type: f_score
      value: 0.9398674862
  - task:
      name: LABELED_DEPENDENCIES
      type: token-classification
    metrics:
    - name: Labeled Attachment Score (LAS)
      type: f_score
      value: 0.9194891243
  - task:
      name: SENTS
      type: token-classification
    metrics:
    - name: Sentences F-Score
      type: f_score
      value: 0.9798617373
---
UD v2.5 benchmarking pipeline for UD_Spanish-AnCora

| Feature | Description |
| --- | --- |
| **Name** | `es_udv25_spanishancora_trf` |
| **Version** | `0.0.1` |
| **spaCy** | `>=3.2.1,<3.3.0` |
| **Default Pipeline** | `experimental_char_ner_tokenizer`, `transformer`, `tagger`, `morphologizer`, `parser`, `experimental_edit_tree_lemmatizer` |
| **Components** | `experimental_char_ner_tokenizer`, `transformer`, `senter`, `tagger`, `morphologizer`, `parser`, `experimental_edit_tree_lemmatizer` |
| **Vectors** | 0 keys, 0 unique vectors (0 dimensions) |
| **Sources** | [Universal Dependencies v2.5](https://lindat.mff.cuni.cz/repository/xmlui/handle/11234/1-3105) (Zeman, Daniel; et al.) |
| **License** | `GNU GPL 3.0` |
| **Author** | [Explosion](https://explosion.ai) |

### Label Scheme

<details>

<summary>View label scheme (2060 labels for 6 components)</summary>

| Component | Labels |
| --- | --- |
| **`experimental_char_ner_tokenizer`** | `TOKEN` |
| **`senter`** | `I`, `S` |
| **`tagger`** | `ADJ`, `ADP`, `ADV`, `AUX`, `AUX_PRON`, `CCONJ`, `DET`, `INTJ`, `NOUN`, `NUM`, `PART`, `PRON`, `PROPN`, `PUNCT`, `PUNCT_VERB_PRON_PUNCT`, `SCONJ`, `SYM`, `VERB`, `VERB_PRON`, `X` |
| **`morphologizer`** | `Definite=Def\|Gender=Masc\|Number=Sing\|POS=DET\|PronType=Art`, `Gender=Masc\|Number=Sing\|POS=NOUN`, `AdpType=Preppron\|POS=ADP`, `Gender=Masc\|Number=Sing\|POS=ADJ`, `AdpType=Prep\|POS=ADP`, `Definite=Def\|Gender=Fem\|Number=Plur\|POS=DET\|PronType=Art`, `POS=PROPN`, `Case=Acc,Dat\|POS=PRON\|Person=3\|PrepCase=Npr\|PronType=Prs\|Reflex=Yes`, `Mood=Ind\|Number=Sing\|POS=VERB\|Person=3\|Tense=Past\|VerbForm=Fin`, `POS=VERB\|VerbForm=Inf`, `Gender=Fem\|Number=Sing\|POS=DET\|PronType=Dem`, `Gender=Fem\|Number=Sing\|POS=NOUN`, `Gender=Fem\|Number=Plur\|POS=NOUN`, `Gender=Fem\|Number=Plur\|POS=DET\|PronType=Ind`, `POS=PRON\|PronType=Int,Rel`, `Mood=Sub\|Number=Plur\|POS=VERB\|Person=3\|Tense=Pres\|VerbForm=Fin`, `Definite=Def\|Gender=Fem\|Number=Sing\|POS=DET\|PronType=Art`, `POS=SCONJ`, `POS=NOUN`, `Definite=Def\|Gender=Masc\|Number=Plur\|POS=DET\|PronType=Art`, `Number=Plur\|POS=NOUN`, `Gender=Masc\|Number=Plur\|POS=DET\|PronType=Ind`, `Gender=Masc\|Number=Plur\|POS=NOUN`, `POS=PUNCT\|PunctType=Peri`, `Mood=Ind\|Number=Sing\|POS=VERB\|Person=3\|Tense=Pres\|VerbForm=Fin`, `POS=PUNCT\|PunctType=Comm`, `Case=Acc\|Gender=Fem\|Number=Sing\|POS=VERB\|Person=3\|PrepCase=Npr\|PronType=Prs\|VerbForm=Inf`, `Mood=Ind\|Number=Plur\|POS=AUX\|Person=3\|Tense=Pres\|VerbForm=Fin`, `Gender=Masc\|Number=Sing\|POS=VERB\|Tense=Past\|VerbForm=Part`, `Number=Plur\|POS=ADJ`, `POS=CCONJ`, `Gender=Masc\|Number=Plur\|POS=PRON\|PronType=Ind`, `POS=ADV`, `Mood=Ind\|Number=Plur\|POS=VERB\|Person=3\|Tense=Fut\|VerbForm=Fin`, `Gender=Masc\|NumType=Card\|Number=Plur\|POS=DET\|PronType=Dem`, `Mood=Ind\|Number=Sing\|POS=AUX\|Person=3\|Tense=Pres\|VerbForm=Fin`, `Number=Sing\|POS=ADJ`, `Gender=Masc\|Number=Plur\|POS=ADJ\|VerbForm=Part`, `Gender=Masc\|Number=Plur\|POS=PRON\|PronType=Tot`, `POS=PRON\|PronType=Ind`, `POS=ADV\|Polarity=Neg`, `Case=Acc\|Gender=Masc\|Number=Sing\|POS=PRON\|Person=3\|PrepCase=Npr\|PronType=Prs`, `Gender=Fem\|Number=Sing\|POS=ADJ`, `Mood=Ind\|Number=Plur\|POS=VERB\|Person=3\|Tense=Past\|VerbForm=Fin`, `Number=Plur\|POS=PRON\|PronType=Int,Rel`, `POS=PUNCT\|PunctType=Quot`, `POS=PUNCT`, `Gender=Masc\|Number=Sing\|POS=ADJ\|VerbForm=Part`, `POS=PUNCT\|PunctSide=Ini\|PunctType=Brck`, `POS=PUNCT\|PunctSide=Fin\|PunctType=Brck`, `NumForm=Digit\|NumType=Card\|POS=NUM`, `NumType=Card\|POS=NUM`, `POS=VERB\|VerbForm=Ger`, `Definite=Ind\|Gender=Masc\|Number=Sing\|POS=DET\|PronType=Art`, `Gender=Masc\|Number=Sing\|POS=DET\|PronType=Dem`, `Gender=Fem\|NumType=Ord\|Number=Plur\|POS=ADJ`, `Number=Sing\|POS=DET\|Person=3\|Poss=Yes\|PronType=Prs`, `Number=Sing\|POS=NOUN`, `Gender=Masc\|Number=Plur\|POS=ADJ`, `Mood=Ind\|Number=Sing\|POS=AUX\|Person=3\|Tense=Fut\|VerbForm=Fin`, `Gender=Fem\|Number=Sing\|POS=ADJ\|VerbForm=Part`, `Mood=Ind\|Number=Plur\|POS=VERB\|Person=3\|Tense=Pres\|VerbForm=Fin`, `Degree=Cmp\|POS=ADV`, `POS=AUX\|VerbForm=Inf`, `Number=Plur\|POS=DET\|PronType=Ind`, `Number=Plur\|POS=DET\|PronType=Dem`, `Degree=Cmp\|Number=Sing\|POS=ADJ`, `Mood=Ind\|Number=Sing\|POS=VERB\|Person=3\|Tense=Fut\|VerbForm=Fin`, `Case=Acc,Dat\|POS=VERB\|Person=3\|PrepCase=Npr\|PronType=Prs\|Reflex=Yes\|VerbForm=Inf`, `Degree=Sup\|Gender=Masc\|Number=Plur\|POS=ADJ`, `Definite=Ind\|Gender=Fem\|Number=Sing\|POS=DET\|PronType=Art`, `AdvType=Tim\|POS=NOUN`, `AdpType=Prep\|POS=ADV`, `Gender=Masc\|Number=Sing\|POS=PRON\|PronType=Ind`, `NumType=Card\|Number=Plur\|POS=NUM`, `AdpType=Preppron\|POS=ADV`, `Case=Acc\|Gender=Masc\|Number=Sing\|POS=VERB\|Person=3\|PrepCase=Npr\|PronType=Prs\|VerbForm=Inf`, `NumForm=Digit\|POS=NOUN`, `Number=Sing\|POS=PRON\|PronType=Dem`, `AdpType=Preppron\|Gender=Masc\|Number=Sing\|POS=ADJ`, `Number=Plur\|POS=DET\|Person=3\|Poss=Yes\|PronType=Prs`, `Gender=Fem\|Number=Plur\|POS=ADJ`, `Gender=Fem\|Number=Plur\|POS=PRON\|PronType=Ind`, `Gender=Masc\|Number=Plur\|POS=DET\|PronType=Tot`, `Mood=Ind\|Number=Sing\|POS=VERB\|Person=3\|Tense=Imp\|VerbForm=Fin`, `Mood=Ind\|Number=Plur\|POS=AUX\|Person=3\|Tense=Past\|VerbForm=Fin`, `Gender=Masc\|Number=Plur\|POS=VERB\|Tense=Past\|VerbForm=Part`, `Gender=Masc\|NumType=Ord\|Number=Sing\|POS=ADJ`, `Gender=Masc\|NumType=Ord\|Number=Plur\|POS=ADJ`, `Gender=Masc\|Number=Plur\|POS=DET\|PronType=Dem`, `Gender=Masc\|Number=Sing\|POS=AUX\|Tense=Past\|VerbForm=Part`, `Number=Sing\|POS=DET\|PronType=Tot`, `Gender=Fem\|Number=Sing\|POS=PRON\|PronType=Ind`, `Mood=Ind\|Number=Sing\|POS=AUX\|Person=1\|Tense=Pres\|VerbForm=Fin`, `Mood=Ind\|Number=Sing\|POS=VERB\|Person=1\|Tense=Pres\|VerbForm=Fin`, `Degree=Cmp\|Number=Plur\|POS=ADJ`, `POS=AUX\|VerbForm=Ger`, `Gender=Fem\|POS=NOUN`, `Gender=Fem\|NumType=Ord\|Number=Sing\|POS=ADJ`, `AdvType=Tim\|POS=ADJ`, `Mood=Ind\|Number=Sing\|POS=AUX\|Person=3\|Tense=Past\|VerbForm=Fin`, `Gender=Fem\|Number=Sing\|POS=VERB\|Tense=Past\|VerbForm=Part`, `Case=Acc\|Gender=Fem\|Number=Sing\|POS=PRON\|Person=3\|PrepCase=Npr\|PronType=Prs`, `Mood=Ind\|Number=Sing\|POS=VERB\|Person=1\|Tense=Imp\|VerbForm=Fin`, `Gender=Fem\|Number=Plur\|POS=ADJ\|VerbForm=Part`, `Gender=Fem\|Number=Plur\|POS=DET\|PronType=Dem`, `Gender=Masc\|Number=Sing\|POS=PRON\|Poss=Yes\|PronType=Int,Rel`, `Number=Sing\|POS=PRON\|PronType=Int,Rel`, `POS=ADJ`, `Mood=Ind\|Number=Sing\|POS=AUX\|Person=1\|Tense=Imp\|VerbForm=Fin`, `Mood=Ind\|Number=Plur\|POS=VERB\|Person=3\|Tense=Imp\|VerbForm=Fin`, `Mood=Ind\|Number=Plur\|POS=AUX\|Person=3\|Tense=Imp\|VerbForm=Fin`, `Mood=Sub\|Number=Sing\|POS=VERB\|Person=3\|Tense=Pres\|VerbForm=Fin`, `Gender=Fem\|Number=Plur\|POS=DET\|PronType=Tot`, `Case=Acc,Nom\|Gender=Masc\|Number=Sing\|POS=PRON\|Person=3\|PronType=Prs`, `Mood=Sub\|Number=Sing\|POS=VERB\|Person=1\|Tense=Pres\|VerbForm=Fin`, `Definite=Ind\|Gender=Fem\|Number=Plur\|POS=DET\|PronType=Art`, `Case=Acc,Nom\|Gender=Fem\|Number=Plur\|POS=PRON\|Person=3\|PronType=Prs`, `Mood=Ind\|Number=Plur\|POS=VERB\|Person=1\|Tense=Pres\|VerbForm=Fin`, `Case=Acc\|Definite=Def\|Gender=Masc\|Number=Sing\|POS=PRON\|Person=3\|PrepCase=Npr\|PronType=Prs`, `Gender=Fem\|Number=Sing\|POS=PRON\|PronType=Dem`, `Mood=Cnd\|Number=Sing\|POS=VERB\|Person=1\|VerbForm=Fin`, `Gender=Masc\|Number=Sing\|POS=DET\|PronType=Tot`, `Number=Plur\|POS=PRON\|PronType=Ind`, `Gender=Masc\|Number=Sing\|POS=DET\|PronType=Ind`, `Case=Dat\|Number=Sing\|POS=PRON\|Person=3\|PronType=Prs`, `POS=PART`, `Gender=Fem\|Number=Sing\|POS=DET\|PronType=Ind`, `Number=Sing\|POS=DET\|PronType=Ind`, `Gender=Masc\|NumType=Card\|Number=Plur\|POS=DET\|PronType=Ind`, `Mood=Cnd\|Number=Plur\|POS=AUX\|Person=3\|VerbForm=Fin`, `NumForm=Digit\|POS=SYM`, `Mood=Imp\|Number=Sing\|POS=VERB\|Person=2\|VerbForm=Fin`, `Case=Dat\|Number=Sing\|POS=VERB\|Person=3\|PronType=Prs\|VerbForm=Inf`, `Gender=Fem\|Number=Plur\|POS=PRON\|PronType=Dem`, `Mood=Cnd\|Number=Sing\|POS=AUX\|Person=1\|VerbForm=Fin`, `NumForm=Digit\|NumType=Frac\|POS=NUM`, `Gender=Fem\|Number=Sing\|POS=PRON\|Poss=Yes\|PronType=Int,Rel`, `Mood=Sub\|Number=Sing\|POS=AUX\|Person=1\|Tense=Pres\|VerbForm=Fin`, `Mood=Sub\|Number=Sing\|POS=VERB\|Person=1\|Tense=Imp\|VerbForm=Fin`, `Gender=Fem\|Number=Sing\|Number[psor]=Plur\|POS=DET\|Person=1\|Poss=Yes\|PronType=Prs`, `Case=Acc,Dat\|Number=Plur\|POS=PRON\|Person=1\|PrepCase=Npr\|PronType=Prs`, `Definite=Ind\|Gender=Masc\|Number=Plur\|POS=DET\|PronType=Art`, `POS=PUNCT\|PunctType=Colo`, `Mood=Sub\|Number=Plur\|POS=AUX\|Person=3\|Tense=Pres\|VerbForm=Fin`, `Mood=Imp\|Number=Plur\|POS=VERB\|Person=3\|VerbForm=Fin`, `Gender=Fem\|Number=Sing\|POS=DET\|PronType=Neg`, `Gender=Masc\|Number=Sing\|POS=PRON\|PronType=Dem`, `Case=Acc\|Gender=Masc\|Number=Plur\|POS=PRON\|Person=3\|PrepCase=Npr\|PronType=Prs`, `Case=Acc\|Gender=Fem\|Number=Plur\|POS=PRON\|Person=3\|PrepCase=Npr\|PronType=Prs`, `Gender=Fem\|Number=Plur\|POS=VERB\|Tense=Past\|VerbForm=Part`, `Case=Acc\|Gender=Fem\|Number=Sing\|POS=AUX\|Person=3\|PrepCase=Npr\|PronType=Prs\|VerbForm=Inf`, `Number=Sing\|POS=PRON\|PronType=Neg`, `POS=PUNCT\|PunctType=Semi`, `Case=Dat\|Number=Plur\|POS=PRON\|Person=3\|PronType=Prs`, `Number=Sing\|POS=PRON\|PronType=Ind`, `Mood=Sub\|Number=Plur\|POS=VERB\|Person=3\|Tense=Imp\|VerbForm=Fin`, `Case=Acc,Nom\|Gender=Masc\|Number=Plur\|POS=PRON\|Person=3\|PronType=Prs`, `POS=INTJ`, `Gender=Masc\|NumType=Card\|Number=Sing\|POS=PRON\|PronType=Dem`, `Mood=Ind\|Number=Plur\|POS=AUX\|Person=3\|Tense=Fut\|VerbForm=Fin`, `Degree=Sup\|Gender=Masc\|Number=Sing\|POS=ADJ`, `Mood=Ind\|Number=Plur\|POS=AUX\|Person=1\|Tense=Pres\|VerbForm=Fin`, `AdpType=Prep\|POS=ADJ`, `Number=Plur\|POS=PRON\|Person=3\|Poss=Yes\|PronType=Prs`, `POS=PUNCT\|PunctType=Dash`, `Mood=Cnd\|Number=Plur\|POS=VERB\|Person=1\|VerbForm=Fin`, `Gender=Masc\|Number=Sing\|POS=DET\|PronType=Neg`, `Gender=Fem\|NumType=Card\|Number=Plur\|POS=NUM`, `Case=Acc\|Gender=Fem\|Number=Plur\|POS=VERB\|Person=3\|PrepCase=Npr\|PronType=Prs\|VerbForm=Inf`, `Gender=Masc\|Number=Sing\|POS=PRON\|PronType=Tot`, `Gender=Masc\|NumType=Card\|Number=Plur\|POS=NUM`, `Gender=Masc\|POS=NOUN`, `Case=Acc,Dat\|Number=Sing\|POS=PRON\|Person=1\|PrepCase=Npr\|PronType=Prs`, `Gender=Fem\|NumType=Card\|Number=Sing\|POS=DET\|PronType=Ind`, `Gender=Fem\|NumType=Card\|Number=Plur\|POS=DET\|PronType=Ind`, `Case=Acc,Dat\|POS=VERB\|Person=3\|PrepCase=Npr\|PronType=Prs\|Reflex=Yes\|VerbForm=Ger`, `Mood=Ind\|Number=Sing\|POS=AUX\|Person=3\|Tense=Imp\|VerbForm=Fin`, `POS=NOUN\|VerbForm=Inf`, `Mood=Ind\|Number=Plur\|POS=AUX\|Person=1\|Tense=Imp\|VerbForm=Fin`, `Mood=Sub\|Number=Sing\|POS=VERB\|Person=3\|Tense=Imp\|VerbForm=Fin`, `Gender=Masc\|Number=Sing\|Number[psor]=Plur\|POS=DET\|Person=1\|Poss=Yes\|PronType=Prs`, `Gender=Masc\|NumType=Card\|Number=Sing\|POS=NUM`, `Mood=Sub\|Number=Sing\|POS=AUX\|Person=1\|Tense=Imp\|VerbForm=Fin`, `Gender=Masc\|Number=Plur\|POS=PRON\|Poss=Yes\|PronType=Int,Rel`, `Case=Acc\|Gender=Masc\|Number=Plur\|POS=VERB\|Person=3\|PrepCase=Npr\|PronType=Prs\|VerbForm=Inf`, `Gender=Fem\|NumType=Card\|Number=Sing\|POS=DET\|PronType=Dem`, `Mood=Imp\|Number=Sing\|POS=VERB\|Person=3\|VerbForm=Fin`, `Mood=Sub\|Number=Plur\|POS=VERB\|Person=1\|Tense=Pres\|VerbForm=Fin`, `Mood=Ind\|Number=Plur\|POS=VERB\|Person=1\|Tense=Fut\|VerbForm=Fin`, `Gender=Masc\|Number=Sing\|POS=PRON\|PronType=Neg`, `Case=Acc,Dat\|Number=Plur\|POS=VERB\|Person=1\|PrepCase=Npr\|PronType=Prs\|VerbForm=Inf`, `Case=Nom\|Number=Sing\|POS=PRON\|Person=1\|PronType=Prs`, `Mood=Ind\|Number=Sing\|POS=VERB\|Person=1\|Tense=Past\|VerbForm=Fin`, `Mood=Ind\|Number=Plur\|POS=VERB\|Person=1\|Tense=Past\|VerbForm=Fin`, `Degree=Abs\|Gender=Masc\|Number=Sing\|POS=ADJ`, `Number=Sing\|Number[psor]=Sing\|POS=DET\|Person=1\|Poss=Yes\|PronType=Prs`, `Case=Acc,Nom\|Gender=Masc\|Number=Plur\|POS=PRON\|Person=1\|PronType=Prs`, `Mood=Imp\|Number=Sing\|POS=AUX\|Person=3\|VerbForm=Fin`, `Mood=Sub\|Number=Sing\|POS=AUX\|Person=3\|Tense=Pres\|VerbForm=Fin`, `Gender=Masc\|Number=Sing\|POS=DET\|Person=3\|Poss=Yes\|PronType=Prs`, `Gender=Fem\|Number=Sing\|POS=DET\|PronType=Tot`, `POS=DET\|PronType=Ind`, `POS=DET\|PronType=Int,Rel`, `AdvType=Tim\|POS=ADV`, `Mood=Cnd\|Number=Sing\|POS=AUX\|Person=3\|VerbForm=Fin`, `POS=PUNCT\|PunctSide=Ini\|PunctType=Qest`, `POS=PUNCT\|PunctSide=Fin\|PunctType=Qest`, `Number=Plur\|Number[psor]=Sing\|POS=DET\|Person=1\|Poss=Yes\|PronType=Prs`, `Gender=Masc\|NumType=Card\|Number=Sing\|POS=DET\|PronType=Ind`, `Mood=Cnd\|Number=Plur\|POS=VERB\|Person=3\|VerbForm=Fin`, `Case=Acc\|Gender=Fem\|Number=Sing\|POS=VERB\|Person=3\|PrepCase=Npr\|PronType=Prs\|VerbForm=Ger`, `Degree=Abs\|Gender=Fem\|Number=Sing\|POS=ADJ`, `Case=Acc,Dat\|Number=Plur\|POS=PRON\|Person=1\|PrepCase=Npr\|PronType=Prs\|Reflex=Yes`, `Mood=Sub\|Number=Plur\|POS=VERB\|Person=1\|Tense=Imp\|VerbForm=Fin`, `Case=Acc,Dat\|Number=Sing\|POS=PRON\|Person=1\|PrepCase=Npr\|PronType=Prs\|Reflex=Yes`, `POS=PUNCT\|PunctSide=Ini\|PunctType=Excl`, `POS=PUNCT\|PunctSide=Fin\|PunctType=Excl`, `Mood=Cnd\|Number=Sing\|POS=VERB\|Person=3\|VerbForm=Fin`, `Case=Acc,Dat\|Mood=Imp\|Number=Sing\|POS=VERB\|Person=3\|PrepCase=Npr\|PronType=Prs\|Reflex=Yes\|VerbForm=Fin`, `Gender=Fem\|Number=Sing\|POS=PRON\|PronType=Tot`, `Gender=Masc\|Number=Plur\|Number[psor]=Plur\|POS=DET\|Person=1\|Poss=Yes\|PronType=Prs`, `Mood=Imp\|Number=Plur\|POS=VERB\|Person=1\|VerbForm=Fin`, `Gender=Masc\|NumType=Card\|Number=Plur\|POS=PRON\|PronType=Ind`, `Gender=Masc\|NumType=Card\|Number=Sing\|POS=PRON\|PronType=Ind`, `Gender=Masc\|Number=Plur\|POS=PRON\|PronType=Dem`, `Case=Dat\|Number=Plur\|POS=VERB\|Person=3\|PronType=Prs\|VerbForm=Inf`, `Degree=Abs\|Gender=Masc\|NumType=Card\|Number=Plur\|POS=DET\|PronType=Ind`, `Case=Acc\|Number=Sing\|POS=PRON\|Person=1\|PrepCase=Pre\|PronType=Prs`, `Case=Acc,Dat\|Mood=Imp\|Number=Plur\|POS=VERB\|Person=3\|PrepCase=Npr\|PronType=Prs\|Reflex=Yes\|VerbForm=Fin`, `Definite=Ind\|Gender=Fem\|NumType=Card\|Number=Sing\|POS=DET\|PronType=Art`, `Gender=Fem\|NumType=Card\|Number=Sing\|POS=NUM`, `Mood=Sub\|Number=Plur\|POS=AUX\|Person=3\|Tense=Imp\|VerbForm=Fin`, `Gender=Fem\|Number=Plur\|Number[psor]=Plur\|POS=DET\|Person=1\|Poss=Yes\|PronType=Prs`, `POS=SCONJ\|PronType=Int,Rel`, `Case=Acc\|POS=PRON\|Person=3\|PrepCase=Pre\|PronType=Prs\|Reflex=Yes`, `Mood=Ind\|Number=Sing\|POS=VERB\|Person=2\|Tense=Pres\|VerbForm=Fin`, `Case=Acc,Dat\|Number=Sing\|POS=VERB\|Person=1\|PrepCase=Npr\|PronType=Prs\|VerbForm=Inf`, `NumType=Card\|Number=Sing\|POS=DET\|PronType=Ind`, `Mood=Ind\|Number=Plur\|POS=VERB\|Person=1\|Tense=Imp\|VerbForm=Fin`, `Case=Acc,Dat\|Number=Sing\|POS=PRON\|Person=2\|PrepCase=Npr\|PronType=Prs`, `Case=Acc,Nom\|Gender=Fem\|Number=Sing\|POS=PRON\|Person=3\|PronType=Prs`, `Number=Sing\|POS=DET\|PronType=Dem`, `Mood=Sub\|Number=Sing\|POS=AUX\|Person=3\|Tense=Imp\|VerbForm=Fin`, `POS=SYM`, `Gender=Fem\|Number=Sing\|POS=PRON\|PronType=Neg`, `Case=Acc\|Gender=Masc\|Number=Sing\|POS=VERB\|Person=3\|PrepCase=Npr\|PronType=Prs\|VerbForm=Ger`, `Degree=Sup\|Gender=Fem\|Number=Sing\|POS=ADJ`, `Case=Nom\|Number=Sing\|POS=PRON\|Person=2\|PronType=Prs`, `Number=Sing\|Number[psor]=Sing\|POS=DET\|Person=2\|Poss=Yes\|PronType=Prs`, `Case=Acc\|Gender=Masc\|Mood=Imp\|Number=Sing\|POS=VERB\|Person=2,3\|PrepCase=Npr\|PronType=Prs\|VerbForm=Fin`, `Mood=Ind\|Number=Sing\|POS=AUX\|Person=1\|Tense=Fut\|VerbForm=Fin`, `Gender=Masc\|Number=Sing\|Number[psor]=Sing\|POS=PRON\|Person=1\|Poss=Yes\|PronType=Ind`, `Case=Acc,Nom\|Number=Sing\|POS=PRON\|Person=2\|Polite=Form\|PronType=Prs`, `Case=Acc\|Gender=Masc\|Number=Plur\|POS=VERB\|Person=3\|PrepCase=Npr\|PronType=Prs\|VerbForm=Ger`, `Gender=Masc\|NumType=Card\|Number=Sing\|POS=PRON\|PronType=Int,Rel`, `Gender=Fem\|NumType=Card\|Number=Plur\|POS=PRON\|PronType=Ind`, `Case=Acc,Dat\|Number=Plur\|POS=VERB\|Person=1\|PrepCase=Npr\|PronType=Prs\|VerbForm=Ger`, `Mood=Ind\|Number=Plur\|POS=VERB\|Person=2\|Tense=Pres\|VerbForm=Fin`, `Mood=Cnd\|Number=Sing\|POS=VERB\|Person=2\|VerbForm=Fin`, `Mood=Ind\|Number=Sing\|POS=VERB\|Person=1\|Tense=Fut\|VerbForm=Fin`, `Mood=Cnd\|Number=Plur\|POS=AUX\|Person=1\|VerbForm=Fin`, `NumType=Card\|Number=Plur\|POS=PRON\|PronType=Ind`, `Gender=Masc\|NumType=Card\|Number=Sing\|POS=DET\|PronType=Dem`, `Degree=Abs\|Gender=Masc\|Number=Sing\|POS=DET\|PronType=Ind`, `Gender=Fem\|Number=Plur\|POS=PRON\|Poss=Yes\|PronType=Int,Rel`, `Mood=Ind\|Number=Sing\|POS=AUX\|Person=1\|Tense=Past\|VerbForm=Fin`, `Case=Acc,Nom\|Number=Plur\|POS=PRON\|Person=2\|Polite=Form\|PronType=Prs`, `Mood=Imp\|Number=Sing\|POS=AUX\|Person=2\|VerbForm=Fin`, `Case=Acc,Dat\|Number=Sing\|POS=VERB\|Person=2\|PrepCase=Npr\|PronType=Prs\|VerbForm=Inf`, `Gender=Fem\|Number=Sing\|Number[psor]=Sing\|POS=PRON\|Person=2\|Poss=Yes\|PronType=Ind`, `NumType=Card\|Number=Sing\|POS=NUM`, `Mood=Ind\|Number=Sing\|POS=VERB\|Person=2\|Tense=Past\|VerbForm=Fin`, `Mood=Ind\|Number=Sing\|POS=AUX\|Person=2\|Tense=Imp\|VerbForm=Fin`, `Mood=Ind\|Number=Sing\|POS=AUX\|Person=2\|Tense=Pres\|VerbForm=Fin`, `Case=Com\|Number=Sing\|POS=PRON\|Person=2\|PrepCase=Pre\|PronType=Prs`, `Mood=Ind\|Number=Sing\|POS=VERB\|Person=2\|Tense=Imp\|VerbForm=Fin`, `Case=Acc,Dat\|Number=Sing\|POS=PRON\|Person=2\|PrepCase=Npr\|PronType=Prs\|Reflex=Yes`, `Case=Acc\|Number=Sing\|POS=PRON\|Person=2\|PrepCase=Pre\|PronType=Prs`, `Mood=Cnd\|Number=Sing\|POS=AUX\|Person=2\|VerbForm=Fin`, `Mood=Sub\|Number=Sing\|POS=AUX\|Person=2\|Tense=Pres\|VerbForm=Fin`, `Number=Sing\|POS=NOUN\|VerbForm=Fin`, `Case=Acc,Dat\|Mood=Imp\|Number=Plur,Sing\|POS=VERB\|Person=1,2\|PrepCase=Npr\|PronType=Prs\|VerbForm=Fin`, `Case=Acc,Dat\|Mood=Imp\|Number=Sing\|POS=VERB\|Person=2\|PrepCase=Npr\|PronType=Prs\|Reflex=Yes\|VerbForm=Fin`, `Mood=Ind\|Number=Sing\|POS=VERB\|Person=2\|Tense=Fut\|VerbForm=Fin`, `Gender=Fem\|NumType=Card\|Number=Sing\|POS=DET\|PronType=Int,Rel`, `Mood=Sub\|Number=Sing\|POS=VERB\|Person=2\|Tense=Pres\|VerbForm=Fin`, `Mood=Ind\|Number=Sing\|POS=AUX\|Person=2\|Tense=Fut\|VerbForm=Fin`, `Gender=Fem\|Number=Plur\|POS=PRON\|PronType=Tot`, `Gender=Masc\|NumType=Card\|Number=Plur\|POS=DET\|PronType=Int,Rel`, `Case=Dat\|Number=Sing\|POS=VERB\|Person=3\|PronType=Prs\|VerbForm=Ger`, `Number=Sing\|POS=VERB\|VerbForm=Fin`, `POS=VERB\|VerbForm=Fin`, `Degree=Abs\|Gender=Masc\|Number=Plur\|POS=ADJ`, `Degree=Abs\|Gender=Fem\|Number=Plur\|POS=ADJ`, `Case=Acc,Dat\|POS=AUX\|Person=3\|PrepCase=Npr\|PronType=Prs\|Reflex=Yes\|VerbForm=Ger`, `Gender=Masc\|Number=Sing\|Number[psor]=Plur\|POS=PRON\|Person=1\|Poss=Yes\|PronType=Prs`, `AdpType=Prep\|Degree=Cmp\|POS=ADV`, `Mood=Sub\|Number=Plur\|POS=AUX\|Person=1\|Tense=Imp\|VerbForm=Fin`, `Gender=Fem\|NumType=Card\|Number=Plur\|POS=DET\|PronType=Dem`, `Definite=Ind\|Gender=Masc\|NumType=Card\|Number=Sing\|POS=DET\|PronType=Art`, `Degree=Sup\|Gender=Fem\|Number=Plur\|POS=ADJ`, `Number=Plur\|POS=PRON\|PronType=Dem`, `Case=Acc,Dat\|Gender=Masc\|Number=Plur\|POS=PRON\|Person=2\|PrepCase=Npr\|PronType=Prs`, `Case=Acc\|Gender=Fem\|Number=Plur\|POS=VERB\|Person=3\|PrepCase=Npr\|PronType=Prs\|VerbForm=Ger`, `Gender=Masc\|Number=Sing\|POS=AUX\|VerbForm=Fin`, `Case=Acc,Dat\|POS=AUX\|Person=3\|PrepCase=Npr\|PronType=Prs\|Reflex=Yes\|VerbForm=Inf`, `Mood=Ind\|Number=Plur\|POS=AUX\|Person=1\|Tense=Past\|VerbForm=Fin`, `Gender=Masc\|NumType=Card\|Number=Sing\|POS=DET\|PronType=Int,Rel`, `Gender=Masc\|Number=Plur\|POS=DET\|Person=3\|Poss=Yes\|PronType=Prs`, `Case=Acc,Dat\|Mood=Imp\|Number=Sing\|POS=VERB\|Person=1,3\|PrepCase=Npr\|PronType=Prs\|VerbForm=Fin`, `Gender=Masc\|NumType=Card\|Number=Plur\|POS=PRON\|PronType=Int,Rel`, `Gender=Masc\|Number=Sing\|POS=PRON\|Person=3\|Poss=Yes\|PronType=Prs`, `Gender=Masc\|Number=Sing\|Number[psor]=Sing\|POS=DET\|Person=1\|Poss=Yes\|PronType=Ind`, `Mood=Ind\|POS=VERB\|Person=3\|Tense=Pres\|VerbForm=Fin`, `Case=Acc,Dat\|Number=Plur\|POS=PRON\|Person=2\|PrepCase=Npr\|PronType=Prs`, `Gender=Masc\|NumType=Card\|Number=Plur\|POS=PRON\|PronType=Dem`, `Gender=Fem\|Number=Sing\|POS=DET\|Person=3\|Poss=Yes\|PronType=Prs`, `Mood=Sub\|Number=Plur\|POS=VERB\|Person=2\|Tense=Pres\|VerbForm=Fin`, `Mood=Sub\|Number=Plur\|POS=AUX\|Person=1\|Tense=Pres\|VerbForm=Fin`, `Mood=Ind\|Number=Plur\|POS=AUX\|Person=1\|Tense=Fut\|VerbForm=Fin`, `Number=Sing\|POS=PRON\|Person=3\|Poss=Yes\|PronType=Prs`, `Case=Acc,Dat\|Number=Sing\|POS=VERB\|Person=2\|PrepCase=Npr\|PronType=Prs\|PunctType=Quot\|VerbForm=Inf`, `Case=Com\|POS=PRON\|Person=3\|PrepCase=Pre\|PronType=Prs\|Reflex=Yes`, `NumForm=Digit\|NumType=Frac\|POS=SYM`, `Case=Dat\|Number=Sing\|POS=AUX\|Person=3\|PronType=Prs\|VerbForm=Inf`, `Case=Acc\|Gender=Masc\|Number=Sing\|POS=AUX\|Person=3\|PrepCase=Npr\|PronType=Prs\|VerbForm=Inf`, `Gender=Fem\|NumType=Card\|Number=Sing\|POS=PRON\|PronType=Ind`, `Gender=Masc\|Mood=Ind\|Number=Sing\|POS=VERB\|Person=3\|Tense=Pres\|VerbForm=Fin`, `Case=Acc,Dat\|Gender=Masc\|Number=Plur\|POS=PRON\|Person=1\|PrepCase=Npr\|PronType=Prs`, `Gender=Fem\|Number=Sing\|Number[psor]=Sing\|POS=PRON\|Person=1\|Poss=Yes\|PronType=Ind`, `Case=Acc,Dat\|Number=Plur\|POS=VERB\|Person=2\|PrepCase=Npr\|PronType=Prs\|VerbForm=Inf`, `Number=Sing\|POS=PRON\|PronType=Tot`, `Mood=Ind\|Number=Plur\|POS=AUX\|Person=2\|Tense=Pres\|VerbForm=Fin`, `Case=Dat\|Number=Plur\|POS=VERB\|Person=3\|PronType=Prs\|VerbForm=Ger`, `NumType=Card\|Number=Plur\|POS=DET\|PronType=Ind`, `POS=PRON\|PronType=Dem`, `POS=AUX\|VerbForm=Fin`, `Gender=Fem\|NumType=Card\|Number=Plur\|POS=PRON\|PronType=Int,Rel`, `Gender=Fem\|Number=Sing\|Number[psor]=Plur\|POS=DET\|Person=2\|Poss=Yes\|PronType=Prs`, `Gender=Fem\|Number=Plur\|Number[psor]=Plur\|POS=PRON\|Person=1\|Poss=Yes\|PronType=Prs`, `Case=Acc\|Gender=Fem\|Number=Plur\|POS=AUX\|Person=3\|PrepCase=Npr\|PronType=Prs\|VerbForm=Inf`, `Case=Acc\|Gender=Masc\|Number=Plur\|POS=AUX\|Person=3\|PrepCase=Npr\|PronType=Prs\|VerbForm=Inf`, `AdvType=Tim\|Gender=Masc\|Number=Sing\|POS=NOUN`, `Gender=Fem\|Number=Sing\|Number[psor]=Plur\|POS=PRON\|Person=1\|Poss=Yes\|PronType=Prs`, `Gender=Fem\|NumType=Card\|Number=Sing\|POS=PRON\|PronType=Dem`, `Gender=Fem\|Number=Sing\|Number[psor]=Sing\|POS=DET\|Person=1\|Poss=Yes\|PronType=Ind`, `Gender=Masc\|Number=Sing\|Number[psor]=Sing\|POS=PRON\|Person=2\|Poss=Yes\|PronType=Ind`, `Gender=Fem\|Number=Plur\|POS=PRON\|Person=3\|Poss=Yes\|PronType=Prs`, `Gender=Masc\|Number=Plur\|POS=DET\|PronType=Art`, `Gender=Masc\|Number=Sing\|POS=NOUN\|VerbForm=Part`, `Case=Acc\|Gender=Masc\|Number=Sing\|POS=AUX\|Person=3\|PrepCase=Npr\|PronType=Prs\|VerbForm=Ger`, `Gender=Masc\|Number=Sing\|POS=DET\|Person=3\|Poss=Yes\|PronType=Ind`, `Case=Acc,Dat\|Number=Sing\|POS=VERB\|Person=1\|PrepCase=Npr\|PronType=Prs\|VerbForm=Ger`, `Case=Acc\|Gender=Masc\|Mood=Imp\|Number=Plur\|POS=VERB\|Person=1,3\|PrepCase=Npr\|PronType=Prs\|VerbForm=Fin`, `Gender=Fem\|Number=Sing\|POS=PRON\|Person=1\|Poss=Yes\|PronType=Prs`, `Case=Com\|Number=Sing\|POS=PRON\|Person=1\|PrepCase=Pre\|PronType=Prs`, `POS=X`, `Case=Com\|POS=PRON\|Person=3\|PronType=Prs\|Reflex=Yes`, `POS=ADP`, `Case=Acc\|Gender=Masc\|Mood=Imp\|Number=Plur,Sing\|POS=VERB\|Person=1,3\|PrepCase=Npr\|PronType=Prs\|VerbForm=Fin`, `Case=Acc,Dat\|Number=Sing\|POS=AUX\|Person=1\|PrepCase=Npr\|PronType=Prs\|VerbForm=Inf`, `Case=Acc\|Gender=Masc\|Mood=Imp\|Number=Sing\|POS=VERB\|Person=3\|PrepCase=Npr\|PronType=Prs\|VerbForm=Fin`, `Mood=Imp\|Number=Plur\|POS=VERB\|Person=2\|VerbForm=Fin`, `Gender=Masc\|Number=Plur\|POS=PRON\|Person=2\|Poss=Yes\|PronType=Ind`, `Case=Dat\|Mood=Imp\|Number=Sing\|POS=VERB\|Person=3\|PronType=Prs\|VerbForm=Fin`, `Gender=Fem\|Number=Sing\|POS=PRON\|Person=3\|Poss=Yes\|PronType=Prs`, `Case=Acc,Dat\|Mood=Imp\|Number=Sing\|POS=VERB\|Person=2,3\|PrepCase=Npr\|PronType=Prs\|VerbForm=Fin`, `Gender=Fem\|Number=Plur\|POS=DET\|Person=3\|Poss=Yes\|PronType=Prs`, `Case=Acc,Dat\|Number=Plur\|POS=VERB\|Person=1\|PrepCase=Npr\|PronType=Prs\|Reflex=Yes\|VerbForm=Ger`, `Gender=Fem\|Number=Plur\|Number[psor]=Plur\|POS=DET\|Person=2\|Poss=Yes\|PronType=Prs`, `Number=Plur\|Number[psor]=Sing\|POS=DET\|Person=2\|Poss=Yes\|PronType=Prs`, `POS=NOUN\|PunctType=Comm`, `Degree=Cmp\|POS=ADJ`, `Gender=Masc\|POS=ADJ`, `Degree=Abs\|Gender=Masc\|NumType=Card\|Number=Plur\|POS=PRON\|PronType=Ind`, `POS=PRON\|PronType=Neg`, `Gender=Fem\|Number=Sing\|POS=DET\|Person=3\|Poss=Yes\|PronType=Ind`, `Number=Sing\|POS=DET\|PronType=Int,Rel` |
| **`parser`** | `ROOT`, `acl`, `advcl`, `advmod`, `amod`, `appos`, `aux`, `aux:pass`, `case`, `cc`, `ccomp`, `compound`, `conj`, `cop`, `csubj`, `dep`, `det`, `expl:pass`, `fixed`, `flat`, `iobj`, `mark`, `nmod`, `nsubj`, `nsubj:pass`, `nummod`, `obj`, `obl`, `orphan`, `parataxis`, `punct`, `xcomp` |
| **`experimental_edit_tree_lemmatizer`** | `1`, `2`, `5`, `6`, `8`, `10`, `14`, `16`, `18`, `20`, `22`, `24`, `25`, `27`, `29`, `33`, `36`, `38`, `40`, `42`, `45`, `48`, `50`, `54`, `57`, `59`, `60`, `62`, `64`, `66`, `68`, `71`, `73`, `75`, `77`, `81`, `83`, `85`, `87`, `88`, `91`, `93`, `95`, `97`, `99`, `100`, `102`, `104`, `106`, `108`, `110`, `112`, `114`, `115`, `117`, `119`, `120`, `122`, `49`, `125`, `126`, `128`, `130`, `134`, `138`, `140`, `143`, `145`, `146`, `148`, `150`, `151`, `153`, `156`, `158`, `160`, `162`, `164`, `167`, `170`, `171`, `173`, `177`, `178`, `179`, `181`, `182`, `184`, `186`, `187`, `188`, `191`, `193`, `195`, `198`, `201`, `202`, `13`, `204`, `206`, `208`, `210`, `214`, `216`, `218`, `221`, `223`, `224`, `226`, `228`, `230`, `232`, `234`, `235`, `237`, `239`, `241`, `242`, `244`, `248`, `250`, `254`, `257`, `258`, `260`, `261`, `262`, `264`, `265`, `266`, `267`, `269`, `271`, `273`, `277`, `278`, `280`, `284`, `286`, `288`, `289`, `290`, `291`, `293`, `296`, `298`, `300`, `302`, `304`, `306`, `308`, `309`, `313`, `315`, `319`, `321`, `322`, `323`, `324`, `325`, `327`, `328`, `330`, `332`, `336`, `338`, `339`, `341`, `342`, `343`, `345`, `347`, `348`, `350`, `351`, `352`, `354`, `355`, `357`, `359`, `361`, `363`, `365`, `367`, `370`, `372`, `375`, `377`, `379`, `382`, `385`, `389`, `391`, `393`, `395`, `397`, `398`, `400`, `402`, `404`, `408`, `410`, `413`, `415`, `416`, `418`, `419`, `420`, `422`, `424`, `427`, `429`, `431`, `433`, `434`, `435`, `436`, `438`, `440`, `441`, `443`, `445`, `447`, `448`, `450`, `451`, `452`, `454`, `456`, `457`, `458`, `460`, `462`, `463`, `465`, `466`, `468`, `470`, `473`, `477`, `478`, `480`, `481`, `483`, `485`, `489`, `491`, `492`, `494`, `496`, `498`, `500`, `501`, `504`, `505`, `506`, `507`, `509`, `511`, `514`, `516`, `519`, `521`, `522`, `524`, `526`, `528`, `532`, `535`, `538`, `541`, `543`, `545`, `546`, `548`, `550`, `554`, `555`, `557`, `559`, `560`, `561`, `562`, `564`, `565`, `567`, `569`, `571`, `572`, `573`, `575`, `576`, `579`, `582`, `584`, `586`, `589`, `590`, `591`, `592`, `595`, `596`, `597`, `599`, `600`, `601`, `603`, `606`, `607`, `608`, `610`, `615`, `617`, `618`, `622`, `624`, `625`, `626`, `627`, `629`, `631`, `633`, `585`, `634`, `636`, `637`, `638`, `639`, `643`, `644`, `646`, `647`, `648`, `650`, `651`, `653`, `654`, `657`, `658`, `660`, `662`, `663`, `667`, `669`, `671`, `673`, `674`, `678`, `680`, `683`, `684`, `685`, `686`, `688`, `689`, `692`, `693`, `695`, `696`, `697`, `699`, `701`, `702`, `704`, `707`, `709`, `711`, `712`, `714`, `715`, `717`, `718`, `719`, `720`, `722`, `725`, `728`, `730`, `732`, `733`, `734`, `735`, `736`, `738`, `739`, `740`, `741`, `743`, `745`, `748`, `750`, `752`, `753`, `755`, `756`, `759`, `760`, `763`, `764`, `765`, `766`, `768`, `770`, `772`, `773`, `774`, `775`, `776`, `778`, `779`, `780`, `783`, `785`, `786`, `788`, `791`, `793`, `795`, `797`, `798`, `800`, `803`, `804`, `805`, `807`, `808`, `810`, `813`, `816`, `819`, `821`, `823`, `824`, `825`, `826`, `829`, `832`, `833`, `836`, `129`, `837`, `838`, `839`, `843`, `845`, `846`, `848`, `849`, `851`, `852`, `853`, `855`, `856`, `857`, `858`, `862`, `864`, `866`, `868`, `869`, `873`, `875`, `877`, `878`, `879`, `882`, `884`, `886`, `888`, `890`, `891`, `892`, `893`, `895`, `897`, `898`, `900`, `902`, `904`, `906`, `907`, `909`, `910`, `912`, `914`, `915`, `916`, `918`, `920`, `921`, `923`, `924`, `926`, `928`, `930`, `931`, `933`, `935`, `936`, `937`, `939`, `940`, `943`, `944`, `945`, `946`, `947`, `949`, `951`, `952`, `953`, `955`, `956`, `957`, `0`, `959`, `961`, `963`, `965`, `966`, `968`, `969`, `970`, `972`, `973`, `975`, `976`, `978`, `979`, `980`, `982`, `983`, `984`, `986`, `987`, `989`, `990`, `993`, `995`, `996`, `997`, `1000`, `1003`, `1004`, `1006`, `1007`, `1008`, `1010`, `1012`, `1013`, `1014`, `1015`, `1017`, `1018`, `1021`, `1025`, `1027`, `1029`, `1030`, `1032`, `1034`, `1035`, `1036`, `1038`, `1039`, `1041`, `1043`, `1044`, `1045`, `1046`, `1047`, `1049`, `1050`, `1052`, `1053`, `1054`, `1055`, `1056`, `1057`, `1058`, `1060`, `1061`, `1063`, `1065`, `1067`, `1069`, `1070`, `1072`, `1075`, `1076`, `1077`, `1078`, `1079`, `1080`, `1081`, `1082`, `1085`, `1086`, `1088`, `1090`, `1091`, `1092`, `1093`, `1094`, `1096`, `1097`, `1100`, `1101`, `1103`, `1104`, `1106`, `1108`, `1109`, `1111`, `1112`, `1114`, `1115`, `1116`, `598`, `26`, `1117`, `1118`, `1119`, `1121`, `1122`, `1123`, `1124`, `1125`, `1127`, `1128`, `1130`, `1132`, `1133`, `1135`, `1137`, `1139`, `1140`, `1141`, `1142`, `1144`, `1147`, `1151`, `1152`, `1153`, `1155`, `1157`, `1160`, `1162`, `1163`, `1165`, `1166`, `1170`, `1171`, `1173`, `1175`, `1177`, `1179`, `1180`, `1183`, `1185`, `1186`, `1188`, `1189`, `1191`, `1192`, `1193`, `1196`, `65`, `1197`, `1198`, `1202`, `1204`, `1206`, `1208`, `1209`, `1210`, `1213`, `1214`, `1215`, `1218`, `1220`, `1221`, `1223`, `1225`, `1226`, `1228`, `1230`, `1232`, `1233`, `1235`, `1236`, `1237`, `1238`, `1241`, `1242`, `1243`, `1244`, `1248`, `1253`, `1254`, `1256`, `1259`, `1260`, `1262`, `1264`, `1265`, `1266`, `1267`, `1269`, `1272`, `1273`, `1274`, `1275`, `1277`, `1280`, `1283`, `1286`, `1289`, `1291`, `1293`, `1294`, `1295`, `1296`, `1297`, `1298`, `1300`, `1301`, `1303`, `1307`, `1309`, `1311`, `1312`, `1316`, `1317`, `1318`, `1319`, `1321`, `1322`, `1323`, `1324`, `1325`, `1326`, `1327`, `1329`, `1330`, `1331`, `1332`, `1333`, `1334`, `1335`, `1336`, `1338`, `1339`, `1341`, `1342`, `1344`, `1346`, `1347`, `1348`, `1349`, `1350`, `1351`, `1352`, `1354`, `1356`, `1357`, `1359`, `1360`, `1361`, `1363`, `1364`, `1365`, `1369`, `1370`, `1371`, `1372`, `1373`, `1377`, `1378`, `1379`, `1381`, `1382`, `1383`, `1385`, `1386`, `1388`, `1389`, `1390`, `1391`, `1392`, `1394`, `1395`, `1396`, `1398`, `1399`, `1400`, `1402`, `1403`, `1406`, `1408`, `1409`, `1410`, `1413`, `1415`, `1416`, `1417`, `1418`, `1419`, `1421`, `1422`, `1423`, `1425`, `1427`, `1428`, `1431`, `1432`, `1433`, `1434`, `1435`, `1437`, `1438`, `1441`, `1442`, `1443`, `1445`, `1446`, `1447`, `1448`, `1449`, `1450`, `1452`, `1453`, `1454`, `1455`, `1457`, `1458`, `1460`, `1462`, `1463`, `1464`, `1467`, `1468`, `1469`, `1470`, `1472`, `1477`, `1479`, `1481`, `1484`, `1486`, `1488`, `1489`, `1492`, `1494`, `1495`, `1496`, `1498`, `1500`, `1501`, `1503`, `1504`, `1505`, `1507`, `1509`, `1510`, `1512`, `1513`, `1514`, `1516`, `1518`, `1519`, `1520`, `1523`, `1525`, `1526`, `1527`, `1529`, `1531`, `1532`, `1533`, `1535`, `1536`, `1537`, `1538`, `1540`, `1541`, `1542`, `1544`, `1546`, `1547`, `1548`, `124`, `1549`, `1551`, `1553`, `1555`, `1557`, `1560`, `1561`, `1563`, `1564`, `1565`, `1569`, `1571`, `1572`, `1573`, `1574`, `1575`, `1577`, `1579`, `1581`, `1582`, `1583`, `1585`, `1588`, `1589`, `1590`, `1591`, `1592`, `1595`, `1596`, `1597`, `1598`, `1599`, `1600`, `1601`, `1603`, `1605`, `1609`, `1611`, `1613`, `1614`, `1618`, `1619`, `1622`, `1624`, `1626`, `1628`, `1630`, `1631`, `1634`, `1636`, `1637`, `1638`, `1640`, `1642`, `1643`, `1644`, `1645`, `1646`, `1648`, `1649`, `1650`, `1651`, `1652`, `1653`, `1654`, `1656`, `1658`, `1660`, `1662`, `1665`, `1667`, `1668`, `1669`, `1671`, `1672`, `1673`, `1674`, `1675`, `1676`, `1678`, `1680`, `1681`, `1682`, `1683`, `1684`, `1685`, `1686`, `1688`, `1689`, `1690`, `1691`, `1692`, `1694`, `1696`, `1697`, `1698`, `1700`, `1701`, `1702`, `1703`, `1704`, `1706`, `1708`, `1709`, `1710`, `1711`, `1712`, `1713`, `1714`, `1715`, `1717`, `1718`, `1719`, `1721`, `1722`, `1724`, `1725`, `1726`, `1728`, `1729`, `1730`, `1731`, `1732`, `1733`, `1735`, `1737`, `1739`, `1741`, `1743`, `1744`, `1745`, `1747`, `1749`, `1750`, `1752`, `1753`, `1756`, `1758`, `1760`, `1761`, `1762`, `1764`, `1765`, `1767`, `1769`, `1772`, `1773`, `1774`, `1775`, `1777`, `1778`, `1781`, `1783`, `1784`, `1786`, `1790`, `1791`, `1792`, `1793`, `1795`, `1796`, `1798`, `1799`, `1801`, `1802`, `1804`, `1805`, `1806`, `1807`, `1809`, `1810`, `1811`, `1814`, `1816`, `1817`, `1818`, `1819`, `1820`, `1822`, `1824`, `1826`, `1827`, `1829`, `1831`, `1832`, `1834`, `1836`, `1838`, `1840`, `1842`, `1843`, `1844`, `1845`, `1847`, `1848`, `1850`, `1851`, `1853`, `1854`, `1856`, `1859`, `1860`, `1861`, `1863`, `1865`, `1866`, `1868`, `1869`, `1870`, `1871`, `1873`, `1875`, `1877`, `1879`, `1881`, `1883`, `1884`, `1887`, `1889`, `1890`, `1892`, `1893`, `1894`, `1895`, `1897`, `1899`, `1902`, `1903`, `1904`, `1906`, `1907`, `1909`, `1910`, `1912`, `1913`, `1914`, `1916`, `1917`, `1918`, `1920`, `1921`, `1923`, `1926`, `1927`, `1928`, `1929`, `1930`, `1931`, `1932`, `1933`, `1934`, `1935`, `1937`, `1938`, `1939`, `1942`, `1943`, `1944`, `1945`, `1946`, `1947`, `1948`, `1949`, `1950`, `1952`, `1953`, `1955`, `1956`, `1957`, `1958`, `1959`, `1961`, `1964`, `1967`, `1969`, `1971`, `1972`, `1974`, `1975`, `1977`, `1978`, `1979`, `1980`, `1981`, `1922`, `1982`, `1983`, `1984`, `1986`, `1988`, `1989`, `1990`, `1992`, `1993`, `1994`, `1995`, `1998`, `1999`, `2000`, `2003`, `2006`, `2007`, `2008`, `2009`, `2011`, `2013`, `2015`, `2016`, `2017`, `2018`, `2020`, `2023`, `2027`, `2028`, `2030`, `2031`, `2032`, `2033`, `2034`, `2035`, `2036`, `2039`, `2042`, `2043`, `2045`, `2047`, `2050`, `2052`, `2053`, `2054`, `2055`, `2056`, `2057`, `2061`, `2062`, `2063`, `2064`, `2065`, `2066`, `2067`, `2068`, `2069`, `2070`, `2073`, `2074`, `2075`, `2076`, `2078`, `2079`, `2080`, `2081`, `2082`, `2083`, `2084`, `2089`, `2090`, `2092`, `2093`, `2094`, `2095`, `2096`, `2098`, `2099`, `2100`, `2101`, `2103`, `2104`, `2106`, `2108`, `2109`, `2110`, `2113`, `2116`, `2119`, `2121`, `2124`, `2125`, `2126`, `2127`, `2128`, `2129`, `2132`, `2133`, `2134`, `2136`, `2137`, `2138`, `2139`, `2140`, `2141`, `2142`, `2143`, `2145`, `2146`, `2147`, `2148`, `2149`, `2150`, `2151`, `2152`, `2153`, `2154`, `2155`, `2157`, `2159`, `2160`, `2161`, `2162`, `2163`, `2164`, `2166`, `2167`, `2169`, `2172`, `2173`, `2174`, `2175`, `2178`, `2180`, `2181`, `2184`, `2186`, `2189`, `2190`, `2191`, `2192`, `2194`, `2195`, `2197`, `2199`, `2200`, `2202`, `2203`, `2204`, `2205`, `2210`, `2211`, `2212`, `2214`, `2215`, `2216`, `2217`, `2218`, `2219`, `2220`, `2221`, `2222`, `2223`, `2225`, `2227`, `2228`, `2229`, `2230`, `2231`, `2232`, `2233`, `2234`, `2235`, `2238`, `2239`, `2240`, `2241`, `2242`, `2243`, `2244`, `2245`, `2246`, `2250`, `2252`, `2254`, `2255`, `2256`, `2257`, `2258`, `2259`, `2260`, `2262`, `2264`, `2265`, `2266`, `2267`, `2268`, `2269`, `2270`, `2271`, `2272`, `2273`, `2274`, `2275`, `2276`, `2277`, `2278`, `2279`, `2280`, `2281`, `2283`, `2284`, `2285`, `2286`, `2287`, `2288`, `2289`, `2290`, `2291`, `2293`, `2294`, `2295`, `2296`, `2297`, `2298`, `2299`, `2301`, `2303`, `2304`, `2305`, `2306`, `2307`, `2308`, `2309`, `2310`, `2312`, `2313`, `2314`, `2315`, `2317`, `2319`, `2320`, `2321`, `2322`, `2324`, `2325`, `2326`, `2328`, `2329`, `2330`, `2331`, `2332`, `2333`, `2334`, `2335`, `2336`, `2337`, `2338`, `2339`, `2341`, `2342`, `2346`, `2347`, `2352`, `2353`, `2356`, `2358`, `2359`, `2360`, `2361`, `2362`, `2364`, `2365`, `2366`, `2368`, `2371`, `2372`, `2374`, `2375`, `2376`, `2377`, `2378`, `2379`, `2380`, `2382`, `2383`, `2384`, `2386`, `2387`, `2388`, `2389`, `2391`, `2394`, `2395`, `2396`, `2398`, `2399`, `2400`, `2401`, `2403`, `2404`, `2406`, `2409`, `2410`, `2411`, `2415`, `2418`, `2419`, `2420`, `2421`, `2422`, `2423`, `2424`, `2425`, `2427`, `430`, `2428`, `2429`, `2430`, `2431`, `2432`, `2433`, `2434`, `2435`, `2436`, `2437`, `2438`, `2439`, `2440`, `2441`, `2442`, `2444`, `2445`, `2446`, `2447`, `2448`, `2449`, `2450`, `2451`, `2452`, `2453`, `2454`, `2456`, `2457`, `2458`, `2460`, `2461`, `2462`, `2463`, `2464`, `2465`, `2466`, `2467`, `2468`, `2469`, `2472`, `2474`, `2475`, `2476`, `2479`, `2480`, `2481`, `2482`, `2483`, `2484`, `2486`, `2487`, `2488`, `2490`, `2491`, `2493`, `2494`, `2495`, `2496`, `2497`, `2499`, `2500`, `2501`, `2502`, `2503`, `2504`, `2505`, `2506`, `2507`, `2508`, `2509`, `2510`, `2511`, `2512`, `2514`, `2515`, `2516`, `2517`, `2518`, `2519`, `2520`, `2521`, `2522`, `2523`, `2524`, `2525`, `2527`, `2528`, `2529`, `2530`, `2531`, `2532`, `2533`, `2535`, `2536`, `2537`, `2538`, `2539`, `2540`, `2541`, `2542`, `2543`, `2544`, `2545`, `2546`, `2547`, `2548`, `2550`, `2552`, `2554`, `2555`, `2556`, `2557`, `2558`, `2559`, `2560`, `2561`, `2562`, `2563`, `2566`, `2567`, `2568`, `2569`, `2570`, `2572`, `2574`, `2576`, `2577`, `2578`, `2580`, `2582`, `2583`, `2584`, `2585` |

</details>

### Accuracy

| Type | Score |
| --- | --- |
| `TOKEN_F` | 99.98 |
| `TOKEN_P` | 99.98 |
| `TOKEN_R` | 99.99 |
| `TOKEN_ACC` | 100.00 |
| `SENTS_F` | 97.99 |
| `SENTS_P` | 97.43 |
| `SENTS_R` | 98.55 |
| `TAG_ACC` | 98.92 |
| `POS_ACC` | 99.03 |
| `MORPH_ACC` | 97.96 |
| `DEP_UAS` | 93.99 |
| `DEP_LAS` | 91.95 |
| `LEMMA_ACC` | 98.93 |
---
tags:
- spacy
- token-classification
language:
- en
model-index:
- name: en_spacy_pii_fast
  results:
  - task:
      name: NER
      type: token-classification
    metrics:
    - name: NER Precision
      type: precision
      value: 0.8682266815
    - name: NER Recall
      type: recall
      value: 0.8475264792
    - name: NER F Score
      type: f_score
      value: 0.8577517086
widget:
- text: "SELECT shipping FROM users WHERE shipping = '201 Thayer St Providence RI 02912'"
---

| Feature | Description |
| --- | --- |
| **Name** | `en_spacy_pii_fast` |
| **Version** | `0.0.0` |
| **spaCy** | `>=3.4.1,<3.5.0` |
| **Default Pipeline** | `tok2vec`, `ner` |
| **Components** | `tok2vec`, `ner` |
| **Vectors** | 0 keys, 0 unique vectors (0 dimensions) |
| **Sources** | n/a |
| **License** | n/a |
| **Author** | [n/a]() |

### Label Scheme

<details>

<summary>View label scheme (5 labels for 1 components)</summary>

| Component | Labels |
| --- | --- |
| **`ner`** | `DATE_TIME`, `LOC`, `NRP`, `ORG`, `PER` |

</details>

### Accuracy

| Type | Score |
| --- | --- |
| `ENTS_F` | 85.78 |
| `ENTS_P` | 86.82 |
| `ENTS_R` | 84.75 |
| `TOK2VEC_LOSS` | 83709.83 |
| `NER_LOSS` | 147916.24 |
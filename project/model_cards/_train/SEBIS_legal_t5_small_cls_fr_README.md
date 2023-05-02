
---
language: French   
tags:
- classification French model
datasets:
- jrc-acquis
widget:
- text: "Règlement (CE) no 264/2005 de la Commission du 16 février 2005 fixant les restitutions à l'exportation dans le secteur de la viande de volaille applicables à partir du 17 février 2005 LA COMMISSION DES COMMUNAUTÉS EUROPÉENNES, vu le traité instituant la Communauté européenne, vu le règlement (CEE) no 2777/75 du Conseil du 29 octobre 1975 portant organisation commune des marchés dans le secteur de la viande de volaille [1], et notamment son article 8, paragraphe 3, troisième alinéa, considérant ce qui suit: (1) Aux termes de l'article 8 du règlement (CEE) no 2777/75, la différence entre les prix des produits visés à l'article 1er, paragraphe 1, dudit règlement, sur le marché mondial et dans la Communauté, peut être couverte par une restitution à l'exportation. (2) L'application de ces règles et critères à la situation actuelle des marchés dans le secteur de la viande de volaille conduit à fixer la restitution à un montant qui permette la participation de la Communauté au commerce international et tienne compte également du caractère des exportations de ces produits ainsi que de leur importance à l'heure actuelle. (3) L'article 21 du règlement (CE) no 800/1999 de la Commission du 15 avril 1999 portant modalités communes d'application du régime des restitutions à l'exportation pour les produits agricoles [2] prévoit qu'aucune restitution n'est octroyée lorsque les produits ne sont pas de qualité saine, loyale et marchande le jour d'acceptation de la déclaration d'exportation. Afin d'assurer une application uniforme de la réglementation en vigueur, il y a lieu de préciser que, pour bénéficier d'une restitution, les viandes de volailles figurant à l'article 1er du règlement (CEE) no 2777/75 doivent porter la marque de salubrité comme prévu à la directive 71/118/CEE du Conseil du 15 février 1971 relative à des problèmes sanitaires en matière de production et de mise sur le marché de viandes fraîches de volaille [3]. (4) Le comité de gestion de la viande de volaille et des œufs n'a pas émis d'avis dans le délai imparti par son président, A ARRÊTÉ LE PRÉSENT RÈGLEMENT: Article premier Les codes des produits pour l'exportation desquels est accordée la restitution visée à l'article 8 du règlement (CEE) no 2777/75 et les montants de cette restitution sont fixés à l'annexe du présent règlement. Toutefois, afin de pouvoir bénéficier de la restitution, les produits entrant dans le champ d'application du chapitre XII de l'annexe de la directive 71/118/CEE doivent également satisfaire aux conditions de marquage de salubrité prévues par cette directive. Article 2 Le présent règlement entre en vigueur le 17 février 2005. Le présent règlement est obligatoire dans tous ses éléments et directement applicable dans tout État membre. Fait à Bruxelles, le 16 février 2005. Par la Commission Mariann Fischer Boel Membre de la Commission [1] JO L 282 du 1.11.1975, p. 77. Règlement modifié en dernier lieu par le règlement (CE) no 806/2003 (JO L 122 du 16.5.2003, p. 1). [2] JO L 102 du 17.4.1999, p. 11. Règlement modifié en dernier lieu par le règlement (CE) no 671/2004 (JO L 105 du 14.4.2004, p. 5). [3] JO L 55 du 8.3.1971, p. 23. Directive modifiée en dernier lieu par le règlement (CE) no 807/2003 (JO L 122 du 16.5.2003, p. 36). -------------------------------------------------- ANNEXE Code des produits | Destination | Unité de mesure | Montant des restitutions | 0105 11 11 9000 | A02 | EUR/100 pcs | 0,80 | 0105 11 19 9000 | A02 | EUR/100 pcs | 0,80 | 0105 11 91 9000 | A02 | EUR/100 pcs | 0,80 | 0105 11 99 9000 | A02 | EUR/100 pcs | 0,80 | 0105 12 00 9000 | A02 | EUR/100 pcs | 1,70 | 0105 19 20 9000 | A02 | EUR/100 pcs | 1,70 | 0207 12 10 9900 | V01 | EUR/100 kg | 41,00 | 0207 12 10 9900 | A24 | EUR/100 kg | 41,00 | 0207 12 90 9190 | V01 | EUR/100 kg | 41,00 | 0207 12 90 9190 | A24 | EUR/100 kg | 41,00 | 0207 12 90 9990 | V01 | EUR/100 kg | 41,00 | 0207 12 90 9990 | A24 | EUR/100 kg | 41,00 | --------------------------------------------------"

---

# legal_t5_small_cls_fr model

Model for classification of legal text written in French. It was first released in
[this repository](https://github.com/agemagician/LegalTrans). This model is trained on three parallel corpus from jrc-acquis.


## Model description

legal_t5_small_cls_fr is based on the `t5-small` model and was trained on a large corpus of parallel text. This is a smaller model, which scales the baseline model of t5 down by using `dmodel = 512`, `dff = 2,048`, 8-headed attention, and only 6 layers each in the encoder and decoder. This variant has about 60 million parameters.

## Intended uses & limitations

The model could be used for classification of legal texts written in French.

### How to use

Here is how to use this model to classify legal text written in French in PyTorch:

```python
from transformers import AutoTokenizer, AutoModelWithLMHead, TranslationPipeline

pipeline = TranslationPipeline(
model=AutoModelWithLMHead.from_pretrained("SEBIS/legal_t5_small_cls_fr"),
tokenizer=AutoTokenizer.from_pretrained(pretrained_model_name_or_path = "SEBIS/legal_t5_small_cls_fr", do_lower_case=False, 
                                            skip_special_tokens=True),
    device=0
)

fr_text = "Règlement (CE) no 264/2005 de la Commission du 16 février 2005 fixant les restitutions à l'exportation dans le secteur de la viande de volaille applicables à partir du 17 février 2005 LA COMMISSION DES COMMUNAUTÉS EUROPÉENNES, vu le traité instituant la Communauté européenne, vu le règlement (CEE) no 2777/75 du Conseil du 29 octobre 1975 portant organisation commune des marchés dans le secteur de la viande de volaille [1], et notamment son article 8, paragraphe 3, troisième alinéa, considérant ce qui suit: (1) Aux termes de l'article 8 du règlement (CEE) no 2777/75, la différence entre les prix des produits visés à l'article 1er, paragraphe 1, dudit règlement, sur le marché mondial et dans la Communauté, peut être couverte par une restitution à l'exportation. (2) L'application de ces règles et critères à la situation actuelle des marchés dans le secteur de la viande de volaille conduit à fixer la restitution à un montant qui permette la participation de la Communauté au commerce international et tienne compte également du caractère des exportations de ces produits ainsi que de leur importance à l'heure actuelle. (3) L'article 21 du règlement (CE) no 800/1999 de la Commission du 15 avril 1999 portant modalités communes d'application du régime des restitutions à l'exportation pour les produits agricoles [2] prévoit qu'aucune restitution n'est octroyée lorsque les produits ne sont pas de qualité saine, loyale et marchande le jour d'acceptation de la déclaration d'exportation. Afin d'assurer une application uniforme de la réglementation en vigueur, il y a lieu de préciser que, pour bénéficier d'une restitution, les viandes de volailles figurant à l'article 1er du règlement (CEE) no 2777/75 doivent porter la marque de salubrité comme prévu à la directive 71/118/CEE du Conseil du 15 février 1971 relative à des problèmes sanitaires en matière de production et de mise sur le marché de viandes fraîches de volaille [3]. (4) Le comité de gestion de la viande de volaille et des œufs n'a pas émis d'avis dans le délai imparti par son président, A ARRÊTÉ LE PRÉSENT RÈGLEMENT: Article premier Les codes des produits pour l'exportation desquels est accordée la restitution visée à l'article 8 du règlement (CEE) no 2777/75 et les montants de cette restitution sont fixés à l'annexe du présent règlement. Toutefois, afin de pouvoir bénéficier de la restitution, les produits entrant dans le champ d'application du chapitre XII de l'annexe de la directive 71/118/CEE doivent également satisfaire aux conditions de marquage de salubrité prévues par cette directive. Article 2 Le présent règlement entre en vigueur le 17 février 2005. Le présent règlement est obligatoire dans tous ses éléments et directement applicable dans tout État membre. Fait à Bruxelles, le 16 février 2005. Par la Commission Mariann Fischer Boel Membre de la Commission [1] JO L 282 du 1.11.1975, p. 77. Règlement modifié en dernier lieu par le règlement (CE) no 806/2003 (JO L 122 du 16.5.2003, p. 1). [2] JO L 102 du 17.4.1999, p. 11. Règlement modifié en dernier lieu par le règlement (CE) no 671/2004 (JO L 105 du 14.4.2004, p. 5). [3] JO L 55 du 8.3.1971, p. 23. Directive modifiée en dernier lieu par le règlement (CE) no 807/2003 (JO L 122 du 16.5.2003, p. 36). -------------------------------------------------- ANNEXE Code des produits | Destination | Unité de mesure | Montant des restitutions | 0105 11 11 9000 | A02 | EUR/100 pcs | 0,80 | 0105 11 19 9000 | A02 | EUR/100 pcs | 0,80 | 0105 11 91 9000 | A02 | EUR/100 pcs | 0,80 | 0105 11 99 9000 | A02 | EUR/100 pcs | 0,80 | 0105 12 00 9000 | A02 | EUR/100 pcs | 1,70 | 0105 19 20 9000 | A02 | EUR/100 pcs | 1,70 | 0207 12 10 9900 | V01 | EUR/100 kg | 41,00 | 0207 12 10 9900 | A24 | EUR/100 kg | 41,00 | 0207 12 90 9190 | V01 | EUR/100 kg | 41,00 | 0207 12 90 9190 | A24 | EUR/100 kg | 41,00 | 0207 12 90 9990 | V01 | EUR/100 kg | 41,00 | 0207 12 90 9990 | A24 | EUR/100 kg | 41,00 | --------------------------------------------------"

pipeline([fr_text], max_length=512)
```

## Training data

The legal_t5_small_cls_fr model was trained on [JRC-ACQUIS](https://wt-public.emm4u.eu/Acquis/index_2.2.html) dataset consisting of 22 Thousand texts.

## Training procedure


The model was trained on a single TPU Pod V3-8 for 250K steps in total, using sequence length 512 (batch size 64). It has a total of approximately 220M parameters and was trained using the encoder-decoder architecture. The optimizer used is AdaFactor with inverse square root learning rate schedule for pre-training.

### Preprocessing

An unigram model trained with 88M lines of text from the parallel corpus (of all possible language pairs) to get the vocabulary (with byte pair encoding), which is used with this model.

### Pretraining



## Evaluation results

When the model is used for classification test dataset, achieves the following results:

Test results :

| Model | F1 score |
|:-----:|:-----:|
|   legal_t5_small_cls_fr | 0.6159|


### BibTeX entry and citation info

> Created by [Ahmed Elnaggar/@Elnaggar_AI](https://twitter.com/Elnaggar_AI) | [LinkedIn](https://www.linkedin.com/in/prof-ahmed-elnaggar/)

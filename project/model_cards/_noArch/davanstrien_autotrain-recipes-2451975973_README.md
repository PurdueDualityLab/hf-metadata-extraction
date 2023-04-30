---
tags:
- autotrain
- text-classification
language:
- en
widget:
- text: "a favourite dutch salad in the way some prime kippered herrings a dn d haddock or some fine yarmouth bloaters then when cold remove all the bones and skin aryl tear the flesh into shreds with two forks sea en these well with pepper salad oil and tarragea vinegar and set aside in a cool place until required cut up into small dice myrtle boil beetroot and potatoes raw cucumber and onions and mix well together with the fish and sonto wellmade tartar sauce then pile up the whols on a flat dish sprinkle well with a mixture of finelychopped parsley and sifted egg yolk garnish round the base with anchovy or saniino crodtons tastefully ornamented with tiny patches or chopped parsley and strips of hardboiled white of egg and servo"
- text: "collieries the men at one of the collieries have in times of scarcity been in the habit houseo f this getting from e v i t a a t t el s eve r a wellnt wishing g h e t r h e a r v e als water disturbed so frequently locked up the well one of the men a blacksmith removed fhe lock and subse quently received notice to leave the colliery the other mechanics decided that unless the the masters at once withdrew the blacksmiths notice they themselves would resign the masters however refused and a fortni"
- text: "made on a certain branch of the fifth nerve sneezing being a redex action excited by saal a slight impression on that nerve sneezing dat s not take place when the fifth nerve is parelyz e even though the sense of smell is retained lentil soupset two quarts of water on to hail with ill red lentils when it has been on an wulf add loz of pearl tapioca that has been provi ns il soaked in a atte cold water salt to taste and ha half an hour longer cost about id another ito is cat into dice a large onion a mediu carrot half as much turnip as carrot oad ga head of celery pat these vegetables tashher b o a pound of lentils into a large saucepan w it h quarts of water and simmer slowly till all the tents are quite soft then pass all through a i f sieve and return to the saucepan with a good of butter and a seasoning of pepper salt e squeeze of lemon ice then boil up drew bide and when quite off the stir in wi im yaks ol ouzoctwa eggs"
datasets:
- davanstrien/autotrain-data-recipes
co2_eq_emissions:
  emissions: 6.990639915807625
---

# Model Trained Using AutoTrain

- Problem type: Multi-class Classification
- Model ID: 2451975973
- CO2 Emissions (in grams): 6.9906

## Validation Metrics

- Loss: 0.046
- Accuracy: 0.989
- Macro F1: 0.936
- Micro F1: 0.989
- Weighted F1: 0.989
- Macro Precision: 0.929
- Micro Precision: 0.989
- Weighted Precision: 0.989
- Macro Recall: 0.943
- Micro Recall: 0.989
- Weighted Recall: 0.989


## Usage


This model has been trained to predict whether an article from a historic newspaper is a 'recipe' or 'not a recipe'. 
This model was trained on data generated by carrying out a keyword search of food terms and annotating examples results to indicate whether they were a recipe. 

You can use cURL to access this model:

```
$ curl -X POST -H "Authorization: Bearer YOUR_API_KEY" -H "Content-Type: application/json" -d '{"inputs": "I love AutoTrain"}' https://api-inference.huggingface.co/models/davanstrien/autotrain-recipes-2451975973
```

Or Python API:

```
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("davanstrien/autotrain-recipes-2451975973", use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained("davanstrien/autotrain-recipes-2451975973", use_auth_token=True)

inputs = tokenizer("I love AutoTrain", return_tensors="pt")

outputs = model(**inputs)
```
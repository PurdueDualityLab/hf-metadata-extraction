---
tags:
- autotrain
- text-classification
language:
- unk
widget:
- text: "I love AutoTrain 🤗"
datasets:
- Ahmed-Abousetta/autotrain-data-abunawaf-information
co2_eq_emissions:
  emissions: 1.5985216080073748
---

# Model Trained Using AutoTrain

- Problem type: Binary Classification
- Model ID: 1859863562
- CO2 Emissions (in grams): 1.5985

## Validation Metrics

- Loss: 0.375
- Accuracy: 0.857
- Precision: 0.832
- Recall: 0.840
- AUC: 0.912
- F1: 0.836

## Usage

You can use cURL to access this model:

```
$ curl -X POST -H "Authorization: Bearer YOUR_API_KEY" -H "Content-Type: application/json" -d '{"inputs": "I love AutoTrain"}' https://api-inference.huggingface.co/models/Ahmed-Abousetta/autotrain-abunawaf-information-1859863562
```

Or Python API:

```
from transformers import AutoModelForSequenceClassification, AutoTokenizer

model = AutoModelForSequenceClassification.from_pretrained("Ahmed-Abousetta/autotrain-abunawaf-information-1859863562", use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained("Ahmed-Abousetta/autotrain-abunawaf-information-1859863562", use_auth_token=True)

inputs = tokenizer("I love AutoTrain", return_tensors="pt")

outputs = model(**inputs)
```
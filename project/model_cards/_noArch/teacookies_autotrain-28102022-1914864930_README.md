---
tags:
- autotrain
- token-classification
language:
- unk
widget:
- text: "I love AutoTrain 🤗"
datasets:
- teacookies/autotrain-data-28102022
co2_eq_emissions:
  emissions: 19.19485186697524
---

# Model Trained Using AutoTrain

- Problem type: Entity Extraction
- Model ID: 1914864930
- CO2 Emissions (in grams): 19.1949

## Validation Metrics

- Loss: 0.002
- Accuracy: 1.000
- Precision: 0.982
- Recall: 0.984
- F1: 0.983

## Usage

You can use cURL to access this model:

```
$ curl -X POST -H "Authorization: Bearer YOUR_API_KEY" -H "Content-Type: application/json" -d '{"inputs": "I love AutoTrain"}' https://api-inference.huggingface.co/models/teacookies/autotrain-28102022-1914864930
```

Or Python API:

```
from transformers import AutoModelForTokenClassification, AutoTokenizer

model = AutoModelForTokenClassification.from_pretrained("teacookies/autotrain-28102022-1914864930", use_auth_token=True)

tokenizer = AutoTokenizer.from_pretrained("teacookies/autotrain-28102022-1914864930", use_auth_token=True)

inputs = tokenizer("I love AutoTrain", return_tensors="pt")

outputs = model(**inputs)
```
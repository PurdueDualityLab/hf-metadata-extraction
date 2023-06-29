---
language: 
- multilingual
- en
- fr
- es
- de
- zh
- ar
- ru
- vi
- el
- bg
- th
- tr
- hi
- ur
- sw
- nl
- uk
- ro
- pt
- it
- lt
- no
- pl
- da 
- ja

datasets: wikipedia

license: apache-2.0

widget:
- text: "Google generated 46 billion [MASK] in revenue."
- text: "Paris is the capital of [MASK]."
- text: "Algiers is the largest city in [MASK]."
- text: "Paris est la [MASK] de la France."
- text: "Paris est la capitale de la [MASK]."
- text: "L'élection américaine a eu [MASK] en novembre 2020."
- text: "تقع سويسرا في [MASK] أوروبا"
- text: "إسمي محمد وأسكن في [MASK]."
---

# distilbert-base-25lang-cased

We are sharing smaller versions of [distilbert-base-multilingual-cased](https://huggingface.co/distilbert-base-multilingual-cased) that handle a custom number of languages.

Our versions give exactly the same representations produced by the original model which preserves the original accuracy.

Handled languages: en, fr, es, de, zh, ar, ru, vi, el, bg, th, tr, hi, ur, sw, nl, uk, ro, pt, it, lt, no, pl, da and ja.

For more information please visit our paper: [Load What You Need: Smaller Versions of Multilingual BERT](https://www.aclweb.org/anthology/2020.sustainlp-1.16.pdf).

## How to use

```python
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("Geotrend/distilbert-base-25lang-cased")
model = AutoModel.from_pretrained("Geotrend/distilbert-base-25lang-cased")

```

To generate other smaller versions of multilingual transformers please visit [our Github repo](https://github.com/Geotrend-research/smaller-transformers).

### How to cite

```bibtex
@inproceedings{smallermbert,
  title={Load What You Need: Smaller Versions of Multilingual BERT},
  author={Abdaoui, Amine and Pradel, Camille and Sigel, Grégoire},
  booktitle={SustaiNLP / EMNLP},
  year={2020}
}
```

## Contact 

Please contact amine@geotrend.fr for any question, feedback or request.
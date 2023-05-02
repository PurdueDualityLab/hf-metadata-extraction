---
language: 
  - zh
license: apache-2.0

tags:
- roberta
- NLU
- NLI
- Chinese

inference: true

widget:
- text: "今天心情不好[SEP]今天很开心"

---

# Erlangshen-Roberta-110M-NLI

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

中文的RoBERTa-wwm-ext-base在数个推理任务微调后的版本。

This is the fine-tuned version of the Chinese RoBERTa-wwm-ext-base model on several NLI datasets.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General  | 自然语言理解 NLU | 二郎神 Erlangshen | Roberta |      110M      |    自然语言推理 NLI     |

## 模型信息 Model Information

基于[chinese-roberta-wwm-ext-base](https://huggingface.co/hfl/chinese-roberta-wwm-ext)，我们在收集的4个中文领域的NLI（自然语言推理）数据集，总计1014787个样本上微调了一个NLI版本。

Based on [chinese-roberta-wwm-ext-base](https://huggingface.co/hfl/chinese-roberta-wwm-ext), we fine-tuned an NLI version on 4 Chinese Natural Language Inference (NLI) datasets, with totaling 1,014,787 samples.

### 下游效果 Performance

|    模型 Model   | cmnli    |  ocnli  | snli    |
| :--------:    | :-----:  | :----:  | :-----:   | 
| Erlangshen-Roberta-110M-NLI | 80.83     |   78.56    | 88.01      |
| Erlangshen-Roberta-330M-NLI | 82.25      |   79.82    | 88      |  
| Erlangshen-MegatronBert-1.3B-NLI | 84.52      |   84.17    | 88.67      |  

## 使用 Usage

``` python
from transformers import BertForSequenceClassification
from transformers import BertTokenizer
import torch

tokenizer=BertTokenizer.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-NLI')
model=BertForSequenceClassification.from_pretrained('IDEA-CCNL/Erlangshen-Roberta-110M-NLI')

texta='今天的饭不好吃'
textb='今天心情不好'

output=model(torch.tensor([tokenizer.encode(texta,textb)]))
print(torch.nn.functional.softmax(output.logits,dim=-1))
```

## 引用 Citation

如果您在您的工作中使用了我们的模型，可以引用我们的[论文](https://arxiv.org/abs/2209.02970)：

If you are using the resource for your work, please cite the our [paper](https://arxiv.org/abs/2209.02970):

```text
@article{fengshenbang,
  author    = {Jiaxing Zhang and Ruyi Gan and Junjie Wang and Yuxiang Zhang and Lin Zhang and Ping Yang and Xinyu Gao and Ziwei Wu and Xiaoqun Dong and Junqing He and Jianheng Zhuo and Qi Yang and Yongfeng Huang and Xiayu Li and Yanghan Wu and Junyu Lu and Xinyu Zhu and Weifeng Chen and Ting Han and Kunhao Pan and Rui Wang and Hao Wang and Xiaojun Wu and Zhongshen Zeng and Chongpei Chen},
  title     = {Fengshenbang 1.0: Being the Foundation of Chinese Cognitive Intelligence},
  journal   = {CoRR},
  volume    = {abs/2209.02970},
  year      = {2022}
}
```

也可以引用我们的[网站](https://github.com/IDEA-CCNL/Fengshenbang-LM/):

You can also cite our [website](https://github.com/IDEA-CCNL/Fengshenbang-LM/):

```text
@misc{Fengshenbang-LM,
  title={Fengshenbang-LM},
  author={IDEA-CCNL},
  year={2021},
  howpublished={\url{https://github.com/IDEA-CCNL/Fengshenbang-LM}},
}
```
---
language: 
  - zh

inference:
  parameters:
    max_new_tokens: 250
    repetition_penalty: 1.1
    top_p: 0.9
    do_sample: True
    
    

license: apache-2.0
---

# Wenzhong2.0-GPT2-3.5B-chinese

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)

## 简介 Brief Introduction

基于悟道数据集预训练，善于处理NLG任务，目前最大的，中文版的GPT2。

Pretraining on Wudao Corpus, focused on handling NLG tasks, the current largest, Chinese GPT2.

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 通用 General  | 自然语言生成 NLG | 闻仲 Wenzhong | GPT2 |      3.5B      |    中文 Chinese     |

## 模型信息 Model Information

为了可以获得一个强大的单向语言模型，我们采用GPT模型结构，并且应用于中文语料上。类似于Wenzhong-GPT2-3.5B，这个模型拥有30层解码器和35亿参数，这比原本的GPT2-XL还要大。不同的是，我们把这个模型在悟道（300G版本）语料上进行预训练。据我们所知，它是目前最大的中文的GPT模型。

To obtain a powerful unidirectional language model, we adopt the GPT model structure and apply it to the Chinese corpus. Similar to Wenzhong-GPT2-3.5B, this model has 30 decoder layers and 3.5 billion parameters, which is larger than the original GPT2-XL. The difference is that we pre-trained this model on the Wudao (300G version) corpus. To the best of our knowledge, it is the largest Chinese GPT model currently available.

## 使用 Usage

### 加载模型 Loading Models

```python 
from transformers import GPT2Tokenizer, GPT2LMHeadModel
tokenizer = GPT2Tokenizer.from_pretrained('IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese')
model = GPT2LMHeadModel.from_pretrained('IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese')
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output = model(**encoded_input)
```

### 使用示例 Usage Examples

```python
from transformers import pipeline, set_seed
set_seed(55)
generator = pipeline('text-generation', model='IDEA-CCNL/Wenzhong2.0-GPT2-3.5B-chinese')
generator("北京位于", max_length=30, num_return_sequences=1)

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

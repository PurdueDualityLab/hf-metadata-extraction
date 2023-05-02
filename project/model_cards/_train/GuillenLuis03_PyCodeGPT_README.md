# PyCodeGPT
A pre-trained GPT model for Python code completion and generation

## What is it?

PyCodeGPT is efficient and effective GPT-Neo-based model for python code generation task, which is similar to [OpenAI Codex](https://openai.com/blog/openai-codex/), [Github Copliot](https://copilot.github.com/), [CodeParrot](https://huggingface.co/blog/codeparrot), [AlphaCode](https://deepmind.com/blog/article/Competitive-programming-with-AlphaCode).

## Training Data
Due to the small size of public released dataset, we proposed to collect data from GitHub from scratch. We first crawled 1.2M python-related repositories hosted by GitHub. Then, we used these repository URLs to download all contents of each repository from GitHub. After that, we got 60M raw python files under 1MB with a total size of 330GB. Finally, we carefully designed various strategies of data cleaning to get about 96GB data for training. Please refer to the following table for the details.

|Model|Repositories|Size and file after filtering|
|:------:|:---:|:---:|
| CodeParrot | 0.56M | 12GB (compressed), 5.4M |
| Codex | 54M | 159GB |
| PyCodeGPT | 1.2M | 96GB, 13M |


## Pretrained models

we aims to train median-large pre-trained models (model size with 110M) based on GPT-Neo:
- PyCodeGPT-110M: derived from GPT-Neo 125M with a vocabulary size of 32K. 

## GitHub

[https://github.com/microsoft/PyCodeGPT](https://github.com/microsoft/PyCodeGPT)

## Evaluation Results

Here's our evaluation result on HumanEval dataset:

Note: our model can have a comparable accuracy with Codex of similar model size.

|Model|Pass@1|Pass@10|Pass@100|
|:------:|:---:|:---:|:---:|
|PyCodeGPT-110M                             |**8.32%**  |**13.53%** |**18.3%**  |
|||||
|GPT-Neo 125M                               |0.75%  |1.88%  |2.97%  |
|GPT-Neo 1.3B                               |4.97%  |7.47%  |16.3%  |
|GPT-Neo 2.7B                               |6.41%  |11.27% |21.37% |
|GPT-J 6B                                   |11.62% |15.74% |27.74% |
|||||
|TabNine                                    |2.58%  |4.35%  |7.59%  |
|||||
|CodeParrot 110M                            |3.80%  |6.57%  |12.78% |
|CodeParrot 1.5B                            |3.58%  |8.03%  |14.96% |
|||||
|Codex 12M                                  |2.00%  |3.62%  |8.58%  |
|Codex 25M                                  |3.21%  |7.1%   |12.89% |
|Codex 42M                                  |5.06%  |8.8%   |15.55% |
|Codex 85M                                  |8.22%  |12.81% |22.4%  |
|Codex 300M                                 |13.17% |20.37% |36.27% |
|Codex 679M                                 |16.22% |25.7%  |40.95% |
|Codex 2.5B                                 |21.36% |35.42% |59.5%  |
|Codex 12B                                  |28.81% |46.81% |72.31% |
|||||
|Pretrained Decoder-only 13M (AlphaCode)    |1.5%   |3.6%   |8.6%   |
|Pretrained Decoder-only 29M (AlphaCode)    |3.4%   |5.8%   |11.2%  |
|Pretrained Decoder-only 55M (AlphaCode)    |4.2%   |8.2%   |16.9%  |
|Pretrained Decoder-only 89M (AlphaCode)    |4.3%   |12.2%  |20.0%  |
|Pretrained Decoder-only 302M (AlphaCode)   |11.6%  |18.8%  |31.8%  |
|Pretrained Decoder-only 685M (AlphaCode)   |14.2%  |24.4%  |38.8%  |
|Pretrained Decoder-only 1.1B (AlphaCode)   |17.1%  |28.2%  |45.3%  |
|||||
|PolyCoder 160M                             |2.13%  |3.35%  |4.88%  |
|PolyCoder 400M                             |2.96%  |5.29%  |11.59% |
|PolyCoder 2.7B                             |5.59%  |9.84%  |17.68% |

## Reference
If you want to use the models, you need to cite our following paper:

```
@inproceedings{CERT,
  title={{CERT}: Continual Pre-training on Sketches for Library-oriented Code Generation},
  author={Zan, Daoguang and Chen, Bei and Yang, Dejian and Lin, Zeqi and Kim, Minsu and Guan, Bei and Wang, Yongji and Chen, Weizhu and Lou, Jian-Guang},
  booktitle={The 2022 International Joint Conference on Artificial Intelligence},
  year={2022}
}
```
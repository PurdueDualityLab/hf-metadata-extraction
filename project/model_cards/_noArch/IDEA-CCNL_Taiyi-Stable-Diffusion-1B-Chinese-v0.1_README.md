---
language: zh
license: creativeml-openrail-m

tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
- zh
- Chinese

inference: true
widget:
- text: "孤帆远影碧空尽，惟见长江天际流,油画"
  example_title: 孤帆远影碧空尽，惟见长江天际流,油画
- text: "日出在印象的港口来回, 唯美, 插画"
  example_title: 日出在印象的港口来回, 唯美, 插画
- text: "科幻, 外星文明, 建筑, 机械感, 4k壁纸"
  example_title: 科幻, 外星文明, 建筑, 机械感, 4k壁纸
- text: "东临碣石, 以观沧海, 波涛汹涌, 插画"
  example_title: 东临碣石, 以观沧海, 波涛汹涌, 插画
- text: "飞流直下三千尺, 疑是银河落九天, 瀑布, 插画"
  example_title: 飞流直下三千尺, 疑是银河落九天, 瀑布, 插画 
- text: "女孩背影, 日落, 唯美插画"
  example_title: 女孩背影, 日落, 唯美插画

  
extra_gated_prompt: |-
  One more step before getting this model.
  This model is open access and available to all, with a CreativeML OpenRAIL-M license further specifying rights and usage.
  The CreativeML OpenRAIL License specifies: 

  1. You can't use the model to deliberately produce nor share illegal or harmful outputs or content 
  2. IDEA-CCNL claims no rights on the outputs you generate, you are free to use them and are accountable for their use which must not go against the provisions set in the license
  3. You may re-distribute the weights and use the model commercially and/or as a service. If you do, please be aware you have to include the same use restrictions as the ones in the license and share a copy of the CreativeML OpenRAIL-M to all your users (please read the license entirely and carefully)
  Please read the full license here: https://huggingface.co/spaces/CompVis/stable-diffusion-license
  
  By clicking on "Access repository" below, you accept that your *contact information* (email address and username) can be shared with the model authors as well.
extra_gated_fields:
 I have read the License and agree with its terms: checkbox
---

# Taiyi-Stable-Diffusion-1B-Chinese-v0.1

- Github: [Fengshenbang-LM](https://github.com/IDEA-CCNL/Fengshenbang-LM)
- Docs: [Fengshenbang-Docs](https://fengshenbang-doc.readthedocs.io/)
- API：[Fengshen-OpenAPI](https://fengshenbang-lm.com/open-api)

## 简介 Brief Introduction

首个开源的中文Stable Diffusion模型，基于0.2亿筛选过的中文图文对训练。

The first open source Chinese Stable diffusion, which was trained on 20M filtered Chinese image-text pairs.

## 在线体验 Gradio Web UI

可以在[Taiyi-Stable-Diffusion-Chinese](https://huggingface.co/spaces/IDEA-CCNL/Taiyi-Stable-Diffusion-Chinese)体验我们的模型。

We support a [Gradio](https://github.com/gradio-app/gradio) Web UI to run Taiyi-Stable-Diffusion-1B-Chinese-v0.1:
[Taiyi-Stable-Diffusion-Chinese](https://huggingface.co/spaces/IDEA-CCNL/Taiyi-Stable-Diffusion-Chinese)

## 简介 Brief Introduction

首个开源的中英双语Stable Diffusion模型，基于0.2亿筛选过的中文图文对训练。

## 模型分类 Model Taxonomy

|  需求 Demand  | 任务 Task       | 系列 Series      | 模型 Model    | 参数 Parameter | 额外 Extra |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| 特殊 Special | 多模态 Multimodal | 太乙 Taiyi | Stable Diffusion |    1B    |     Chinese     |

## 模型信息 Model Information

我们将[Noah-Wukong](https://wukong-dataset.github.io/wukong-dataset/)数据集(100M)和[Zero](https://zero.so.com/)数据集(23M)用作预训练的数据集，先用[IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese)对这两个数据集的图文对相似性进行打分，取CLIP Score大于0.2的图文对作为我们的训练集。 我们使用[IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese)作为初始化的text encoder，冻住[stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)([论文](https://arxiv.org/abs/2112.10752))模型的其他部分，只训练text encoder，以便保留原始模型的生成能力且实现中文概念的对齐。该模型目前在0.2亿图文对上训练了一个epoch。 我们在 32 x A100 训练了大约100小时。该版本只是一个初步的版本，我们将持续优化并开源后续模型，欢迎交流。

We use [Noah-Wukong](https://wukong-dataset.github.io/wukong-dataset/)(100M) 和 [Zero](https://zero.so.com/)(23M) as our dataset, and take the image and text pairs with CLIP Score (based on [IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese)) greater than 0.2 as our Training set. We use [IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese](https://huggingface.co/IDEA-CCNL/Taiyi-CLIP-RoBERTa-102M-ViT-L-Chinese) as our init text encoder. To keep the powerful generative capability of stable diffusion and align Chinese concepts with the images, We only train the text encoder and freeze other part of the [stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4)([paper](https://arxiv.org/abs/2112.10752)) model. It takes 100 hours to train this model based on 32 x A100. This model is a preliminary version and we will update this model continuously and open sourse. Welcome to exchange！

### Result
Basic Prompt

|  铁马冰河入梦来，3D绘画。   |  飞流直下三千尺，油画。 | 女孩背影，日落，唯美插画。  |
|  ----  | ----  | ----  |
| ![](result_examples/tiema.png)  | ![](result_examples/feiliu.png)  | ![](result_examples/nvhai.jpg) |

Advanced Prompt

| 铁马冰河入梦来，概念画，科幻，玄幻，3D  | 中国海边城市，科幻，未来感，唯美，插画。 | 那人却在灯火阑珊处，色彩艳丽，古风，资深插画师作品，桌面高清壁纸。 |
|  ----  | ----  | ----  |
| ![](result_examples/tiema2.jpg)  | ![](result_examples/chengshi.jpg) | ![](result_examples/naren.jpg) |


## 使用 Usage

### 全精度 Full precision

```py
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained("IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1").to("cuda")

prompt = '飞流直下三千尺，油画'
image = pipe(prompt, guidance_scale=7.5).images[0]  
image.save("飞流.png")
```

### 半精度 Half precision FP16 (CUDA)

添加 `torch_dtype=torch.float16` 和 `device_map="auto"` 可以快速加载 FP16 的权重，以加快推理速度。
更多信息见 [the optimization docs](https://huggingface.co/docs/diffusers/main/en/optimization/fp16#half-precision-weights)。

```py
# !pip install git+https://github.com/huggingface/accelerate
import torch
from diffusers import StableDiffusionPipeline
torch.backends.cudnn.benchmark = True
pipe = StableDiffusionPipeline.from_pretrained("IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1", torch_dtype=torch.float16)
pipe.to('cuda')

prompt = '飞流直下三千尺，油画'
image = pipe(prompt, guidance_scale=7.5).images[0]  
image.save("飞流.png")
```

### 使用手册 Handbook for Taiyi

https://github.com/IDEA-CCNL/Fengshenbang-LM/blob/main/fengshen/examples/stable_diffusion_chinese/taiyi_handbook.md

### 怎样微调 How to finetune

https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/examples/finetune_taiyi_stable_diffusion

### webui配置 Configure webui

https://github.com/IDEA-CCNL/stable-diffusion-webui/blob/master/README.md

### DreamBooth

https://github.com/IDEA-CCNL/Fengshenbang-LM/tree/main/fengshen/examples/stable_diffusion_dreambooth

## 引用 Citation

如果您在您的工作中使用了我们的模型，可以引用我们的[总论文](https://arxiv.org/abs/2209.02970)：

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


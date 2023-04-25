---
tags:
- image-classification
- timm
library_tag: timm
license: cc-by-nc-4.0
datasets:
- imagenet-1k
- imagenet-1k
---
# Model card for convnextv2_atto.fcmae_ft_in1k

A ConvNeXt-V2 image classification model. Pretrained with a fully convolutional masked autoencoder framework (FCMAE) and fine-tuned on ImageNet-1k.

## Model Details
- **Model Type:** Image classification / feature backbone
- **Model Stats:**
  - Params (M): 3.7
  - GMACs: 0.6
  - Activations (M): 3.8
  - Image size: train = 224 x 224, test = 288 x 288
- **Papers:**
  - ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders: https://arxiv.org/abs/2301.00808
- **Original:** https://github.com/facebookresearch/ConvNeXt-V2
- **Dataset:** ImageNet-1k
- **Pretrain Dataset:** ImageNet-1k

## Model Usage
### Image Classification
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model('convnextv2_atto.fcmae_ft_in1k', pretrained=True)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

top5_probabilities, top5_class_indices = torch.topk(output.softmax(dim=1) * 100, k=5)
```

### Feature Map Extraction
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'convnextv2_atto.fcmae_ft_in1k',
    pretrained=True,
    features_only=True,
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # unsqueeze single image into batch of 1

for o in output:
    # print shape of each feature map in output
    # e.g.:
    #  torch.Size([1, 40, 56, 56])
    #  torch.Size([1, 80, 28, 28])
    #  torch.Size([1, 160, 14, 14])
    #  torch.Size([1, 320, 7, 7])

    print(o.shape)
```

### Image Embeddings
```python
from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'convnextv2_atto.fcmae_ft_in1k',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model = model.eval()

# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model)
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor

# or equivalently (without needing to set num_classes=0)

output = model.forward_features(transforms(img).unsqueeze(0))
# output is unpooled, a (1, 320, 7, 7) shaped tensor

output = model.forward_head(output, pre_logits=True)
# output is a (1, num_features) shaped tensor
```

## Model Comparison
Explore the dataset and runtime metrics of this model in timm [model results](https://github.com/huggingface/pytorch-image-models/tree/main/results).

All timing numbers from eager model PyTorch 1.13 on RTX 3090 w/ AMP.

| model                                                                                                                        |top1  |top5  |img_size|param_count|gmacs |macts |samples_per_sec|batch_size|
|------------------------------------------------------------------------------------------------------------------------------|------|------|--------|-----------|------|------|---------------|----------|
| [convnextv2_huge.fcmae_ft_in22k_in1k_512](https://huggingface.co/timm/convnextv2_huge.fcmae_ft_in22k_in1k_512)               |88.848|98.742|512     |660.29     |600.81|413.07|28.58          |48        |
| [convnextv2_huge.fcmae_ft_in22k_in1k_384](https://huggingface.co/timm/convnextv2_huge.fcmae_ft_in22k_in1k_384)               |88.668|98.738|384     |660.29     |337.96|232.35|50.56          |64        |
| [convnext_xxlarge.clip_laion2b_soup_ft_in1k](https://huggingface.co/timm/convnext_xxlarge.clip_laion2b_soup_ft_in1k)         |88.612|98.704|256     |846.47     |198.09|124.45|122.45         |256       |
| [convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384](https://huggingface.co/timm/convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384)             |88.312|98.578|384     |200.13     |101.11|126.74|196.84         |256       |
| [convnextv2_large.fcmae_ft_in22k_in1k_384](https://huggingface.co/timm/convnextv2_large.fcmae_ft_in22k_in1k_384)             |88.196|98.532|384     |197.96     |101.1 |126.74|128.94         |128       |
| [convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_320](https://huggingface.co/timm/convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_320)             |87.968|98.47 |320     |200.13     |70.21 |88.02 |283.42         |256       |
| [convnext_xlarge.fb_in22k_ft_in1k_384](https://huggingface.co/timm/convnext_xlarge.fb_in22k_ft_in1k_384)                     |87.75 |98.556|384     |350.2      |179.2 |168.99|124.85         |192       |
| [convnextv2_base.fcmae_ft_in22k_in1k_384](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k_384)               |87.646|98.422|384     |88.72      |45.21 |84.49 |209.51         |256       |
| [convnext_large.fb_in22k_ft_in1k_384](https://huggingface.co/timm/convnext_large.fb_in22k_ft_in1k_384)                       |87.476|98.382|384     |197.77     |101.1 |126.74|194.66         |256       |
| [convnext_large_mlp.clip_laion2b_augreg_ft_in1k](https://huggingface.co/timm/convnext_large_mlp.clip_laion2b_augreg_ft_in1k) |87.344|98.218|256     |200.13     |44.94 |56.33 |438.08         |256       |
| [convnextv2_large.fcmae_ft_in22k_in1k](https://huggingface.co/timm/convnextv2_large.fcmae_ft_in22k_in1k)                     |87.26 |98.248|224     |197.96     |34.4  |43.13 |376.84         |256       |
| [convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384](https://huggingface.co/timm/convnext_base.clip_laion2b_augreg_ft_in12k_in1k_384)                   |87.138|98.212|384     |88.59      |45.21 |84.49 |365.47         |256       |
| [convnext_xlarge.fb_in22k_ft_in1k](https://huggingface.co/timm/convnext_xlarge.fb_in22k_ft_in1k)                             |87.002|98.208|224     |350.2      |60.98 |57.5  |368.01         |256       |
| [convnext_base.fb_in22k_ft_in1k_384](https://huggingface.co/timm/convnext_base.fb_in22k_ft_in1k_384)                         |86.796|98.264|384     |88.59      |45.21 |84.49 |366.54         |256       |
| [convnextv2_base.fcmae_ft_in22k_in1k](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in22k_in1k)                       |86.74 |98.022|224     |88.72      |15.38 |28.75 |624.23         |256       |
| [convnext_large.fb_in22k_ft_in1k](https://huggingface.co/timm/convnext_large.fb_in22k_ft_in1k)                               |86.636|98.028|224     |197.77     |34.4  |43.13 |581.43         |256       |
| [convnext_base.clip_laiona_augreg_ft_in1k_384](https://huggingface.co/timm/convnext_base.clip_laiona_augreg_ft_in1k_384)     |86.504|97.97 |384     |88.59      |45.21 |84.49 |368.14         |256       |
| [convnext_base.clip_laion2b_augreg_ft_in12k_in1k](https://huggingface.co/timm/convnext_base.clip_laion2b_augreg_ft_in12k_in1k)                           |86.344|97.97 |256     |88.59      |20.09 |37.55 |816.14         |256       |
| [convnextv2_huge.fcmae_ft_in1k](https://huggingface.co/timm/convnextv2_huge.fcmae_ft_in1k)                                   |86.256|97.75 |224     |660.29     |115.0 |79.07 |154.72         |256       |
| [convnext_small.in12k_ft_in1k_384](https://huggingface.co/timm/convnext_small.in12k_ft_in1k_384)                             |86.182|97.92 |384     |50.22      |25.58 |63.37 |516.19         |256       |
| [convnext_base.clip_laion2b_augreg_ft_in1k](https://huggingface.co/timm/convnext_base.clip_laion2b_augreg_ft_in1k)           |86.154|97.68 |256     |88.59      |20.09 |37.55 |819.86         |256       |
| [convnext_base.fb_in22k_ft_in1k](https://huggingface.co/timm/convnext_base.fb_in22k_ft_in1k)                                 |85.822|97.866|224     |88.59      |15.38 |28.75 |1037.66        |256       |
| [convnext_small.fb_in22k_ft_in1k_384](https://huggingface.co/timm/convnext_small.fb_in22k_ft_in1k_384)                       |85.778|97.886|384     |50.22      |25.58 |63.37 |518.95         |256       |
| [convnextv2_large.fcmae_ft_in1k](https://huggingface.co/timm/convnextv2_large.fcmae_ft_in1k)                                 |85.742|97.584|224     |197.96     |34.4  |43.13 |375.23         |256       |
| [convnext_small.in12k_ft_in1k](https://huggingface.co/timm/convnext_small.in12k_ft_in1k)                                     |85.174|97.506|224     |50.22      |8.71  |21.56 |1474.31        |256       |
| [convnext_tiny.in12k_ft_in1k_384](https://huggingface.co/timm/convnext_tiny.in12k_ft_in1k_384)                               |85.118|97.608|384     |28.59      |13.14 |39.48 |856.76         |256       |
| [convnextv2_tiny.fcmae_ft_in22k_in1k_384](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k_384)               |85.112|97.63 |384     |28.64      |13.14 |39.48 |491.32         |256       |
| [convnextv2_base.fcmae_ft_in1k](https://huggingface.co/timm/convnextv2_base.fcmae_ft_in1k)                                   |84.874|97.09 |224     |88.72      |15.38 |28.75 |625.33         |256       |
| [convnext_small.fb_in22k_ft_in1k](https://huggingface.co/timm/convnext_small.fb_in22k_ft_in1k)                               |84.562|97.394|224     |50.22      |8.71  |21.56 |1478.29        |256       |
| [convnext_large.fb_in1k](https://huggingface.co/timm/convnext_large.fb_in1k)                                                 |84.282|96.892|224     |197.77     |34.4  |43.13 |584.28         |256       |
| [convnext_tiny.in12k_ft_in1k](https://huggingface.co/timm/convnext_tiny.in12k_ft_in1k)                                       |84.186|97.124|224     |28.59      |4.47  |13.44 |2433.7         |256       |
| [convnext_tiny.fb_in22k_ft_in1k_384](https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k_384)                         |84.084|97.14 |384     |28.59      |13.14 |39.48 |862.95         |256       |
| [convnextv2_tiny.fcmae_ft_in22k_in1k](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k)                       |83.894|96.964|224     |28.64      |4.47  |13.44 |1452.72        |256       |
| [convnext_base.fb_in1k](https://huggingface.co/timm/convnext_base.fb_in1k)                                                   |83.82 |96.746|224     |88.59      |15.38 |28.75 |1054.0         |256       |
| [convnextv2_nano.fcmae_ft_in22k_in1k_384](https://huggingface.co/timm/convnextv2_nano.fcmae_ft_in22k_in1k_384)               |83.37 |96.742|384     |15.62      |7.22  |24.61 |801.72         |256       |
| [convnext_small.fb_in1k](https://huggingface.co/timm/convnext_small.fb_in1k)                                                 |83.142|96.434|224     |50.22      |8.71  |21.56 |1464.0         |256       |
| [convnextv2_tiny.fcmae_ft_in1k](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in1k)                                   |82.92 |96.284|224     |28.64      |4.47  |13.44 |1425.62        |256       |
| [convnext_tiny.fb_in22k_ft_in1k](https://huggingface.co/timm/convnext_tiny.fb_in22k_ft_in1k)                                 |82.898|96.616|224     |28.59      |4.47  |13.44 |2480.88        |256       |
| [convnext_nano.in12k_ft_in1k](https://huggingface.co/timm/convnext_nano.in12k_ft_in1k)                                       |82.282|96.344|224     |15.59      |2.46  |8.37  |3926.52        |256       |
| [convnext_tiny_hnf.a2h_in1k](https://huggingface.co/timm/convnext_tiny_hnf.a2h_in1k)                                         |82.216|95.852|224     |28.59      |4.47  |13.44 |2529.75        |256       |
| [convnext_tiny.fb_in1k](https://huggingface.co/timm/convnext_tiny.fb_in1k)                                                   |82.066|95.854|224     |28.59      |4.47  |13.44 |2346.26        |256       |
| [convnextv2_nano.fcmae_ft_in22k_in1k](https://huggingface.co/timm/convnextv2_nano.fcmae_ft_in22k_in1k)                       |82.03 |96.166|224     |15.62      |2.46  |8.37  |2300.18        |256       |
| [convnextv2_nano.fcmae_ft_in1k](https://huggingface.co/timm/convnextv2_nano.fcmae_ft_in1k)                                   |81.83 |95.738|224     |15.62      |2.46  |8.37  |2321.48        |256       |
| [convnext_nano_ols.d1h_in1k](https://huggingface.co/timm/convnext_nano_ols.d1h_in1k)                                         |80.866|95.246|224     |15.65      |2.65  |9.38  |3523.85        |256       |
| [convnext_nano.d1h_in1k](https://huggingface.co/timm/convnext_nano.d1h_in1k)                                                 |80.768|95.334|224     |15.59      |2.46  |8.37  |3915.58        |256       |
| [convnextv2_pico.fcmae_ft_in1k](https://huggingface.co/timm/convnextv2_pico.fcmae_ft_in1k)                                   |80.304|95.072|224     |9.07       |1.37  |6.1   |3274.57        |256       |
| [convnext_pico.d1_in1k](https://huggingface.co/timm/convnext_pico.d1_in1k)                                                   |79.526|94.558|224     |9.05       |1.37  |6.1   |5686.88        |256       |
| [convnext_pico_ols.d1_in1k](https://huggingface.co/timm/convnext_pico_ols.d1_in1k)                                           |79.522|94.692|224     |9.06       |1.43  |6.5   |5422.46        |256       |
| [convnextv2_femto.fcmae_ft_in1k](https://huggingface.co/timm/convnextv2_femto.fcmae_ft_in1k)                                 |78.488|93.98 |224     |5.23       |0.79  |4.57  |4264.2         |256       |
| [convnext_femto_ols.d1_in1k](https://huggingface.co/timm/convnext_femto_ols.d1_in1k)                                         |77.86 |93.83 |224     |5.23       |0.82  |4.87  |6910.6         |256       |
| [convnext_femto.d1_in1k](https://huggingface.co/timm/convnext_femto.d1_in1k)                                                 |77.454|93.68 |224     |5.22       |0.79  |4.57  |7189.92        |256       |
| [convnextv2_atto.fcmae_ft_in1k](https://huggingface.co/timm/convnextv2_atto.fcmae_ft_in1k)                                   |76.664|93.044|224     |3.71       |0.55  |3.81  |4728.91        |256       |
| [convnext_atto_ols.a2_in1k](https://huggingface.co/timm/convnext_atto_ols.a2_in1k)                                           |75.88 |92.846|224     |3.7        |0.58  |4.11  |7963.16        |256       |
| [convnext_atto.d2_in1k](https://huggingface.co/timm/convnext_atto.d2_in1k)                                                   |75.664|92.9  |224     |3.7        |0.55  |3.81  |8439.22        |256       |

## Citation
```bibtex
@article{Woo2023ConvNeXtV2,
  title={ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders},
  author={Sanghyun Woo, Shoubhik Debnath, Ronghang Hu, Xinlei Chen, Zhuang Liu, In So Kweon and Saining Xie},
  year={2023},
  journal={arXiv preprint arXiv:2301.00808},
}
```
```bibtex
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/huggingface/pytorch-image-models}}
}
```

---
license: creativeml-openrail-m
tags:
- pytorch
- diffusers
- stable-diffusion
- text-to-image
- diffusion-models-class
- dreambooth-hackathon
- wildcard
datasets: llhbr/dreamboot
widget:
- text: a photo of cpg-products beer on a bar
---

# DreamBooth model for the cpg-products concept trained by llhbr on the llhbr/dreamboot dataset.

This is a Stable Diffusion model fine-tuned on the cpg-products concept with DreamBooth. It can be used by modifying the `instance_prompt`: **a photo of cpg-products beer**

This model was created as part of the DreamBooth Hackathon 🔥. Visit the [organisation page](https://huggingface.co/dreambooth-hackathon) for instructions on how to take part!

## Description


This is a Stable Diffusion model fine-tuned on `beer` images for the wildcard theme.


## Usage

```python
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained('llhbr/cpg-products-beer')
image = pipeline().images[0]
image
```

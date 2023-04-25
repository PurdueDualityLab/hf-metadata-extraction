---
license: creativeml-openrail-m
tags:
- pytorch
- diffusers
- stable-diffusion
- text-to-image
- diffusion-models-class
- dreambooth-hackathon
- animal
widget:
- text: a dashdash cat fight with alian in Loch Ness
---

# DreamBooth model for the dashdash concept trained by jiaenyue.

This is a Stable Diffusion model fine-tuned on the dashdash concept with DreamBooth. It can be used by modifying the `instance_prompt`: **a photo of dashdash cat**

This model was created as part of the DreamBooth Hackathon 🔥. Visit the [organisation page](https://huggingface.co/dreambooth-hackathon) for instructions on how to take part!

## Description


This is a Stable Diffusion model fine-tuned on `cat` images for the animal theme, 
for the Hugging Face DreamBooth Hackathon, from the HF CN Community, 
corporated with the HeyWhale.


## Usage

```python
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained('jiaenyue/dashdash-cat-heywhale')
image = pipeline().images[0]
image
```

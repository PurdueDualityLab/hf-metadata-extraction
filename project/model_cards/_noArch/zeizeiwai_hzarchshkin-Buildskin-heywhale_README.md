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
widget:
- text: The building skin of the office building, the glass curtain wall
---

# DreamBooth model for the hzarchshkin concept trained by zeizeiwai.

This is a Stable Diffusion model fine-tuned on the hzarchshkin concept with DreamBooth. It can be used by modifying the `instance_prompt`: **a photo of hzarchshkin Buildskin**

This model was created as part of the DreamBooth Hackathon 🔥. Visit the [organisation page](https://huggingface.co/dreambooth-hackathon) for instructions on how to take part!

## Description


This is a Stable Diffusion model fine-tuned on `Buildskin` images for the wildcard theme, 
for the Hugging Face DreamBooth Hackathon, from the HF CN Community, 
corporated with the HeyWhale.


## Usage

```python
from diffusers import StableDiffusionPipeline

pipeline = StableDiffusionPipeline.from_pretrained('zeizeiwai/hzarchshkin-Buildskin-heywhale')
image = pipeline().images[0]
image
```
---
license: creativeml-openrail-m
language: 
  - en
thumbnail: "https://huggingface.co/Norod78/sd2-simpsons-blip/raw/main/example/sd2-simpsons-blip-sample_tile_resized.jpg"
tags:
- stable-diffusion
- stable-diffusion-diffusers
- text-to-image
datasets:
- Norod78/simpsons-blip-captions
inference: true
---

# Simpsons diffusion v2.0
*Stable Diffusion v2.0 fine tuned on images related to "The Simpsons"

If you want more details on how to generate your own blip cpationed dataset see this [colab](https://colab.research.google.com/gist/Norod/ee6ee3c4bf11c2d2be531d728ec30824/buildimagedatasetwithblipcaptionsanduploadtohf.ipynb)

Training was done using a slightly modified version of Hugging-Face's text to image training [example script](https://github.com/huggingface/diffusers/blob/main/examples/text_to_image/train_text_to_image.py)

## About

Put in a text prompt and generate cartoony/simpsony images

## AUTOMATIC1111 webui checkpoint

The [main](https://huggingface.co/Norod78/sd2-simpsons-blip/tree/main) folder contains a .ckpt and a .yaml file to be put in [stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) "stable-diffusion-webui/models/Stable-diffusion" folder and used to generate images

## Sample code


```py
from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler
import torch

# this will substitute the default PNDM scheduler for K-LMS  
lms = LMSDiscreteScheduler(
    beta_start=0.00085, 
    beta_end=0.012, 
    beta_schedule="scaled_linear"
)

guidance_scale=8.5
seed=777
steps=50

cartoon_model_path = "Norod78/sd2-simpsons-blip"
cartoon_pipe = StableDiffusionPipeline.from_pretrained(cartoon_model_path, scheduler=lms, torch_dtype=torch.float16)
cartoon_pipe.to("cuda")

def generate(prompt, file_prefix ,samples):
    torch.manual_seed(seed)
    prompt += ", Very detailed, clean, high quality, sharp image"
    cartoon_images = cartoon_pipe([prompt] * samples, num_inference_steps=steps, guidance_scale=guidance_scale)["images"]
    for idx, image in enumerate(cartoon_images):
        image.save(f"{file_prefix}-{idx}-{seed}-sd2-simpsons-blip.jpg")

generate("An oil painting of Snoop Dogg as a simpsons character", "01_SnoopDog", 4)
generate("Gal Gadot, cartoon", "02_GalGadot", 4)
generate("A cartoony Simpsons town", "03_SimpsonsTown", 4)
generate("Pikachu with the Simpsons, Eric Wallis", "04_PikachuSimpsons", 4)
```

![Images generated by this sample code](https://huggingface.co/Norod78/sd2-simpsons-blip/resolve/main/example/sd2-simpsons-blip-sample_tile_resized.jpg)

## Dataset and Training

Finetuned for 10,000 iterations upon [stabilityai/stable-diffusion-2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base) on [BLIP captioned Simpsons images](https://huggingface.co/datasets/Norod78/simpsons-blip-captions) using 1xA5000 GPU on my home desktop computer

Trained by [@Norod78](https://twitter.com/Norod78)
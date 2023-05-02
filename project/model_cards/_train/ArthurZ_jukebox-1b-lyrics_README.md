---
tags:
- MusicGeneration
- jukebox
---

<!--Copyright 2020 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except in compliance with
the License. You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on
an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License for the
specific language governing permissions and limitations under the License.
-->

# Jukebox

## Overview

The Jukebox model was proposed in [Jukebox: A generative model for music](https://arxiv.org/pdf/2005.00341.pdf)
by Prafulla Dhariwal, Heewoo Jun, Christine Payne, Jong Wook Kim, Alec Radford,
Ilya Sutskever.

This model proposes a generative music model which can be produce minute long samples which can bne conditionned on
artist, genre and lyrics.

The abstract from the paper is the following:

We introduce Jukebox, a model that generates
music with singing in the raw audio domain. We
tackle the long context of raw audio using a multiscale VQ-VAE to compress it to discrete codes,
and modeling those using autoregressive Transformers. We show that the combined model at
scale can generate high-fidelity and diverse songs
with coherence up to multiple minutes. We can
condition on artist and genre to steer the musical
and vocal style, and on unaligned lyrics to make
the singing more controllable. We are releasing
thousands of non cherry-picked samples, along
with model weights and code.

Tips:

This model is very slow for now, and takes 18h to generate a minute long audio. 

This model was contributed by [Arthur Zucker](https://huggingface.co/ArthurZ).
The original code can be found [here](https://github.com/openai/jukebox).

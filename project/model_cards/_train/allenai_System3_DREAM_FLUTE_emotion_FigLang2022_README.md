---
language: "en"  # Example: en
license: "cc-by-4.0"  # Example: apache-2.0 or any license from https://hf.co/docs/hub/repositories-licenses
library_name: "transformers"  # Optional. Example: keras or any library from https://github.com/huggingface/hub-docs/blob/main/js/src/lib/interfaces/Libraries.ts
---
# Model description
This is the T5-3B model for System 3 DREAM-FLUTE (emotion), as described in our paper Just-DREAM-about-it: Figurative Language Understanding with DREAM-FLUTE, FigLang workshop @ EMNLP 2022 (Arxiv link: https://arxiv.org/abs/2210.16407) 

Systems 3: DREAM-FLUTE - Providing DREAM’s different dimensions as input context 

We adapt DREAM’s scene elaborations (Gu et al., 2022) for the figurative language understanding NLI task by using the DREAM model to generate elaborations for the premise and hypothesis separately. This allows us to investigate if similarities or differences in the scene elaborations for the premise and hypothesis will provide useful signals for entailment/contradiction label prediction and improving explanation quality. The input-output format is:
```
Input <Premise> <Premise-elaboration-from-DREAM> <Hypothesis> <Hypothesis-elaboration-from-DREAM>
Output <Label> <Explanation>
```
where the scene elaboration dimensions from DREAM are: consequence, emotion, motivation, and social norm. We also consider a system incorporating all these dimensions as additional context.

In this model, DREAM-FLUTE (emotion), we use elaborations along the "emotion" dimension. For more details on DREAM, please refer to DREAM: Improving Situational QA by First Elaborating the Situation, NAACL 2022 (Arxiv link: https://arxiv.org/abs/2112.08656, ACL Anthology link: https://aclanthology.org/2022.naacl-main.82/).

# How to use this model?
We provide a quick example of how you can try out DREAM-FLUTE (emotion) in our paper with just a few lines of code:
```
>>> from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
>>> model = AutoModelForSeq2SeqLM.from_pretrained("allenai/System3_DREAM_FLUTE_emotion_FigLang2022")

>>> tokenizer = AutoTokenizer.from_pretrained("t5-3b")
>>> input_string = "Premise: we laid in the field of green grass and relaxed. [Premise - emotion] I (myself)'s emotion is happy. Hypothesis: we laid in fields of gold. [Hypothesis - emotion] I (myself)'s emotion is happy. Is there a contradiction or entailment between the premise and hypothesis?"
>>> input_ids = tokenizer.encode(input_string, return_tensors="pt")
>>> output = model.generate(input_ids, max_length=200)
>>> tokenizer.batch_decode(output, skip_special_tokens=True)
['Answer : Entailment. Explanation : Gold is a color that is associated with happiness, so the fields of gold are associated with happiness.']
```

# More details about DREAM-FLUTE ...
For more details about DREAM-FLUTE, please refer to our:
* 📄Paper: https://arxiv.org/abs/2210.16407 
* 💻GitHub Repo: https://github.com/allenai/dream/ 

This model is part of our DREAM-series of works. This is a line of research where we make use of scene elaboration for building a "mental model" of situation given in text. Check out our GitHub Repo for more!

# More details about this model ...
## Training and evaluation data

We use the FLUTE dataset for the FigLang2022SharedTask (https://huggingface.co/datasets/ColumbiaNLP/FLUTE) for training this model. ∼7500 samples are provided as the training set. We used a 80-20 split to create our own training (6027 samples) and validation (1507 samples) partitions on which we build our models. For details on how we make use of the training data provided in the FigLang2022 shared task, please refer to https://github.com/allenai/dream/blob/main/FigLang2022SharedTask/Process_Data_Train_Dev_split.ipynb.

## Model details
This model is a fine-tuned version of [t5-3b](https://huggingface.co/t5-3b).

It achieves the following results on the evaluation set:
- Loss: 0.7557
- Rouge1: 58.5894
- Rouge2: 38.6
- Rougel: 52.5083
- Rougelsum: 52.4698
- Gen Len: 40.5607

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 1
- eval_batch_size: 1
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2
- total_train_batch_size: 2
- total_eval_batch_size: 2
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Training results

| Training Loss | Epoch | Step | Validation Loss | Rouge1  | Rouge2  | Rougel  | Rougelsum | Gen Len |
|:-------------:|:-----:|:----:|:---------------:|:-------:|:-------:|:-------:|:---------:|:-------:|
| 0.9974        | 0.33  | 1000 | 0.8938          | 39.8909 | 27.4849 | 38.2724 | 38.2772   | 18.9987 |
| 0.8991        | 0.66  | 2000 | 0.8294          | 41.2504 | 29.3637 | 39.4768 | 39.478    | 18.9987 |
| 0.8778        | 1.0   | 3000 | 0.7886          | 41.3175 | 29.7998 | 39.5926 | 39.5752   | 19.0    |
| 0.5592        | 1.33  | 4000 | 0.7973          | 41.0529 | 30.2234 | 39.5836 | 39.5931   | 19.0    |
| 0.5608        | 1.66  | 5000 | 0.7784          | 41.6251 | 30.6274 | 40.0233 | 39.9929   | 19.0    |
| 0.5433        | 1.99  | 6000 | 0.7557          | 41.8485 | 30.7651 | 40.3159 | 40.2707   | 19.0    |
| 0.3363        | 2.32  | 7000 | 0.8384          | 41.4456 | 30.8368 | 39.9368 | 39.9349   | 19.0    |
| 0.3434        | 2.65  | 8000 | 0.8529          | 41.7845 | 31.3056 | 40.3295 | 40.339    | 18.9920 |
| 0.3548        | 2.99  | 9000 | 0.8310          | 41.9755 | 31.601  | 40.4929 | 40.5058   | 18.9954 |


### Framework versions

- Transformers 4.22.0.dev0
- Pytorch 1.12.1+cu113
- Datasets 2.4.0
- Tokenizers 0.12.1

---
language: 
  - en
  - code
  
license: "mit"

tags:
- Diff Model
- pytorch
- causal-lm
- code-generation
- The Pile
---

**Model Description**

Diff-Codegen-350M is the first in a series of diff models released by CarperAI. A diff model is an autoregressive language model trained on edits to a piece of text, formatted in Unified Diff Format. These diff models can suggest, given a section of text and a description of the desired change, an intelligent change to the text that fits the description, marking the lines added, changed, and deleted in diff format. The primary use case for these models is for suggesting changes to code—as such, most models we release will be fine-tuned versions of models trained on code datasets.

Diff-Codegen-350M-v1 is an initial preliminary release of an experimental artifact and should be treated as such. We are releasing these results and this model in the hopes that it may be useful to the greater research community, especially those interested in LMs for code.

CarperAI will be releasing larger diff LMs trained on larger code datasets in the near future, building on this initial release.

**Training Data**

This model is a fine-tune of Codegen-350m-mono by Salesforce. This language model was first pre-trained on The PIle, an 800Gb dataset composed of varied web corpora. The datasheet and paper for the Pile can be found here and here respectively. The model was then fine-tuned on a large corpus of code data in multiple languages, before finally being fine-tuned on a Python code dataset. The Codegen paper with full details of these datasets can be found here.

Our diff model was trained on a dataset of commits from BigQuery, a large-scale dataset of many programming languages from GitHub repositories. We filtered the dataset by the number of stars in the repository (>100 stars), license (only open-source non-copyleft licensed code included), and length of file (files greater than 2048 tokens in length were excluded).

The model was trained using the GPT-2 tokenizer.

**Training Details**

The model was trained for 44574 steps (1 epoch) on 8 A100 GPUs.

Each file was formatted as follows for input to the language model:

```
<NME> {FILE_NAME}
<BEF> {INPUT_FILE}
<MSG> {COMMIT_MESSAGE}
<DFF> {FILE_DIFF}
```

**Intended Uses and Limitations**

Due to the model’s small size and restriction to code, one should not expect the model to generalize to domains beyond code and perform (successful) reasoning over large chunks of code. This model is intended to be used in prototyping ELM-like systems, and for solely experimental purposes. This model is provided without warranty and should not be used in commercial settings -- even though the license permits.

**Limitations and Biases**

Due to the short context length restriction and due to the fact that all repositories with under 100 stars were excluded, we expect our diff model to underperform on underrepresented languages, for instance Lean or Coq.

The output of this model should not be trusted as correct and secure code. This model should not be used in any mission critical setting where security is of importance. Similarly, when running the output of this model, it should be done in a sandbox like gVisor. 

**Evaluation Results**

Since this model was trained for prototyping, no evaluation has been performed. Future releases will have extensive evaluation.

**Licensing**

This model is licensed as MIT. While it can be used in commercial settings, we do not recommend its use in commercial settings.


**Acknowledgements**

We’d like to thank Honglu Fan, Harry Saini, Herbie Bradley, and Joel Lehman
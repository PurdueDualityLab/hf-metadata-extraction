---
language: en
datasets:
- common_voice
- mozilla-foundation/common_voice_6_0
metrics:
- wer
- cer
tags:
- audio
- automatic-speech-recognition
- en
- hf-asr-leaderboard
- mozilla-foundation/common_voice_6_0
- robust-speech-event
- speech
- xlsr-fine-tuning-week
license: apache-2.0
model-index:
- name: XLSR Wav2Vec2 English by Jonatas Grosman
  results:
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Common Voice en
      type: common_voice
      args: en
    metrics:
    - name: Test WER
      type: wer
      value: 19.06
    - name: Test CER
      type: cer
      value: 7.69
    - name: Test WER (+LM)
      type: wer
      value: 14.81
    - name: Test CER (+LM)
      type: cer
      value: 6.84
  - task:
      name: Automatic Speech Recognition
      type: automatic-speech-recognition
    dataset:
      name: Robust Speech Event - Dev Data
      type: speech-recognition-community-v2/dev_data
      args: en
    metrics:
    - name: Dev WER
      type: wer
      value: 27.72
    - name: Dev CER
      type: cer
      value: 11.65
    - name: Dev WER (+LM)
      type: wer
      value: 20.85
    - name: Dev CER (+LM)
      type: cer
      value: 11.01
---

# Wav2Vec2-Large-XLSR-53-English

Fine-tuned [facebook/wav2vec2-large-xlsr-53](https://huggingface.co/facebook/wav2vec2-large-xlsr-53) on English using the [Common Voice](https://huggingface.co/datasets/common_voice).
When using this model, make sure that your speech input is sampled at 16kHz.

This model has been fine-tuned thanks to the GPU credits generously given by the [OVHcloud](https://www.ovhcloud.com/en/public-cloud/ai-training/) :)

The script used for training can be found here: https://github.com/jonatasgrosman/wav2vec2-sprint

## Usage

The model can be used directly (without a language model) as follows...

Using the [HuggingSound](https://github.com/jonatasgrosman/huggingsound) library:

```python
from huggingsound import SpeechRecognitionModel

model = SpeechRecognitionModel("jonatasgrosman/wav2vec2-large-xlsr-53-english")
audio_paths = ["/path/to/file.mp3", "/path/to/another_file.wav"]

transcriptions = model.transcribe(audio_paths)
```

Writing your own inference script:

```python
import torch
import librosa
from datasets import load_dataset
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

LANG_ID = "en"
MODEL_ID = "jonatasgrosman/wav2vec2-large-xlsr-53-english"
SAMPLES = 10

test_dataset = load_dataset("common_voice", LANG_ID, split=f"test[:{SAMPLES}]")

processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)

# Preprocessing the datasets.
# We need to read the audio files as arrays
def speech_file_to_array_fn(batch):
    speech_array, sampling_rate = librosa.load(batch["path"], sr=16_000)
    batch["speech"] = speech_array
    batch["sentence"] = batch["sentence"].upper()
    return batch

test_dataset = test_dataset.map(speech_file_to_array_fn)
inputs = processor(test_dataset["speech"], sampling_rate=16_000, return_tensors="pt", padding=True)

with torch.no_grad():
    logits = model(inputs.input_values, attention_mask=inputs.attention_mask).logits

predicted_ids = torch.argmax(logits, dim=-1)
predicted_sentences = processor.batch_decode(predicted_ids)

for i, predicted_sentence in enumerate(predicted_sentences):
    print("-" * 100)
    print("Reference:", test_dataset[i]["sentence"])
    print("Prediction:", predicted_sentence)
```

| Reference  | Prediction |
| ------------- | ------------- |
| "SHE'LL BE ALL RIGHT." | SHE'LL BE ALL RIGHT |
| SIX | SIX |
| "ALL'S WELL THAT ENDS WELL." | ALL AS WELL THAT ENDS WELL |
| DO YOU MEAN IT? | DO YOU MEAN IT |
| THE NEW PATCH IS LESS INVASIVE THAN THE OLD ONE, BUT STILL CAUSES REGRESSIONS. | THE NEW PATCH IS LESS INVASIVE THAN THE OLD ONE BUT STILL CAUSES REGRESSION |
| HOW IS MOZILLA GOING TO HANDLE AMBIGUITIES LIKE QUEUE AND CUE? | HOW IS MOSLILLAR GOING TO HANDLE ANDBEWOOTH HIS LIKE Q AND Q |
| "I GUESS YOU MUST THINK I'M KINDA BATTY." | RUSTIAN WASTIN PAN ONTE BATTLY |
| NO ONE NEAR THE REMOTE MACHINE YOU COULD RING? | NO ONE NEAR THE REMOTE MACHINE YOU COULD RING |
| SAUCE FOR THE GOOSE IS SAUCE FOR THE GANDER. | SAUCE FOR THE GUICE IS SAUCE FOR THE GONDER |
| GROVES STARTED WRITING SONGS WHEN SHE WAS FOUR YEARS OLD. | GRAFS STARTED WRITING SONGS WHEN SHE WAS FOUR YEARS OLD |

## Evaluation

1. To evaluate on `mozilla-foundation/common_voice_6_0` with split `test`

```bash
python eval.py --model_id jonatasgrosman/wav2vec2-large-xlsr-53-english --dataset mozilla-foundation/common_voice_6_0 --config en --split test
```

2. To evaluate on `speech-recognition-community-v2/dev_data`

```bash
python eval.py --model_id jonatasgrosman/wav2vec2-large-xlsr-53-english --dataset speech-recognition-community-v2/dev_data --config en --split validation --chunk_length_s 5.0 --stride_length_s 1.0
```

## Citation
If you want to cite this model you can use this:

```bibtex
@misc{grosman2021wav2vec2-large-xlsr-53-english,
  title={XLSR Wav2Vec2 English by Jonatas Grosman},
  author={Grosman, Jonatas},
  publisher={Hugging Face},
  journal={Hugging Face Hub},
  howpublished={\url{https://huggingface.co/jonatasgrosman/wav2vec2-large-xlsr-53-english}},
  year={2021}
}
```
import sqlite3
from huggingface_hub import HfApi
import random

# Hugging Face Hub API init
hf_api = HfApi(
    endpoint="https://huggingface.co",
    token="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"
)

# Connect to PeaTMOSS sql db
con = sqlite3.connect("PeaTMOSS.db")
cur = con.cursor()

# Choose from model database
cur.execute("SELECT * FROM model")

# init special categories
nlp = []
cv = []
audio = []
tabular = []
rl = []

# Go through each row in model database
for row in cur:

    # repo_url
    url = str(row[4])

    # downloads
    try:
        downloads = int(row[5])
    except:
        downloads = 0
        continue

    # id
    id = str(row[1])

    # All URLs from huggingface with downloads over 1000
    if url.startswith("https://huggingface.co") and downloads >= 1000:

        # query each model's tag names and classify them based on tags present
        try:
            tags = hf_api.model_info(id).tags
            if "tabular" in tags or "table-to-text" in tags:
                tabular += [url]
            elif "deep-reinforcement-learning" in tags or "Stable-Baselines3" in tags or "Reinforcement Learning" in tags:
                rl += [url]
            elif "computer-vision" in tags or "Object Detection" in tags:
                cv += [url]
            elif "audio" in tags or "audio-to-audio" in tags or "audio-classification" in tags or "audio-spectrogram-transformer" in tags or "audio processing" in tags or "audio-frame-classification" tags or "pyannote-audio-pipeline" in tags or "text-to-audio" in tags:
                audio += [url]
            elif "NLP" in tags or "text-generation" in tags:
                nlp += [url]
        except:
            continue

# Random sample of 10 models from each special category
if len(nlp) >= 10:
    print("Natural Language Processing:")
    print(random.sample(nlp, 10))
else:
    print("Natural Language Processing:")
    print(nlp)
if len(cv) >= 10:
    print("Computer Vision:")
    print(random.sample(cv, 10))
else:
    print("Computer Vision:")
    print(cv)
if len(audio) >= 10:
    print("Audio:")
    print(random.sample(audio, 10))
else:
    print("Audio:")
    print(audio)
if len(tabular) >= 10:
    print("Tabular:")
    print(random.sample(tabular, 10))
else:
    print("Tabular:")
    print(tabular)
if len(rl) >= 10:
    print("Reinforcement Learning:")
    print(random.sample(tabular, 10))
else:
    print("Reinforcement Learning:")
    print(rl)

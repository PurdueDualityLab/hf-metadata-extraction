import huggingface_hub
from huggingface_hub import HfApi, list_models, ModelCard
from collections import Counter
import time
import keys
import openai
import json
import tokenizer
import dataCleansing
import argparse
import socket
import asyncio

start_time = time.time()

MAX_RETRIES = 1000
TIME_OUT_ERROR_COUNT = 0

parser = argparse.ArgumentParser(description="Python Script with Different Modes")
parser.add_argument("--mode", choices=["name_only", "card_only", "both"], required=True, help="Select the mode")
args = parser.parse_args()

hf_api = HfApi(
    endpoint="https://huggingface.co", # Can be a Private Hub endpoint.
    token= keys.HUGGINGFACE_API_KEY, # Token is not persisted on the machine.
)
openai.api_key = keys.OPENAI_API_KEY

def load_card(model_name):
    card = ModelCard.load(model_name)
    return card.content
 

def contextualDataFeeding(model_name):
    chatlog = []
    token_count = 0
    card = load_card(model_name)
    subsections = dataCleansing.split_by_subsections(card)
    chatlog.append({"role" : "system", "content" : "extract metadata from given AI model, the output should only be one or two words, if information not present return null \n"})
    # token count = 58
    token_count += 58
    if args.mode == "name_only" or args.mode == "both":
        chatlog.append({"role": "user", "content" : "this model is"}) 
        # token count = 16
        token_count += 16
        chatlog.append({"role": "assistant", "content" : model_name + " on huggingface"})
        token_count += tokenizer.num_tokens_in_chat({"role": "assistant", "content" : model_name + " on huggingface"})

    if args.mode == "card_only" or args.mode == "both":
        for section_headers in subsections:
            chatlog.append({"role": "user", "content" : "the model" + section_headers})
            token_count += tokenizer.num_tokens_in_chat({"role": "user", "content" : "the model" + section_headers})
            chatlog.append({"role" : "assistant", "content" : subsections[section_headers]})
            token_count += tokenizer.num_tokens_in_chat({"role" : "assistant", "content" : subsections[section_headers]})
            if token_count > 4096:
                temp = chatlog.pop()
                temp = chatlog.pop()
                break
            #print(subsections[section_headers])
    return chatlog

async def async_chat(chatlog):
    chat = openai.ChatCompletion.create(
            model = "gpt-3.5-turbo",
            messages = chatlog
        )
    return chat

async def metadataExtraction(chatlog):
    chatlog.append({"role": "user", "content" : "What model is this model fine tuned on, if not fine tuned on any other model return null. Answer concisely with only the name of the model"})

    output = []
    for i in range(5):
        while True:
            try:
                chat = await asyncio.wait_for(async_chat(chatlog), timeout= 10)
                break
            except TimeoutError:
                TIME_OUT_ERROR_COUNT += 1
                print(f"timeout error {TIME_OUT_ERROR_COUNT}\n")
            except openai.error.APIError:
                time.sleep(10)
                print(f"API gateway error \n`")
            except openai.error.RateLimitError:
                time.sleep(10)
        output.append(chat['choices'][0]['message']['content'])
        print(f"resoponse{i}: {chat['choices'][0]['message']['content']}")

    string_count = Counter(output)
    most_common = string_count.most_common(1)
    #print("\n" + most_common[0][0])

    return most_common[0][0]

with open("filtered_models.json", 'r') as json_file:
    data = json.load(json_file)

results = {architecture: {} for architecture in list(data.keys())}

retry_count = 0
for architecture in list(data.keys()):
    for model in data[architecture]:
        while retry_count < MAX_RETRIES:
            try:
                chatlog = contextualDataFeeding(model)
                print("context done")
                data_extracted = asyncio.run(metadataExtraction(chatlog))
                print("extract done\n")
                results[architecture][model]=data_extracted
                print(f"{model} : {data_extracted}\n")
                break
            except openai.error.Timeout:
                retry_count += 1
                print(f"retry count from timeout: {retry_count}\n")
            except openai.error.ServiceUnavailableError:
                retry_count += 1
                print(f"retry count from service unavilable: {retry_count}\n")
            except ValueError:
                results[architecture][model] = "Invalid 'model_index' in" + str(model) + ", model card could not be parsed"
                print(f"Invalid 'model_index' in {model}, model card could not be parsed\n")
                break
            except huggingface_hub.utils._errors.RepositoryNotFoundError:
                results[architecture][model] = str(model) + "repo not found"
                print(f"{model} repo not found\n")
                break
            except Exception as e:
                results[architecture][model] = "An exception of type" + type(e).__name__ + "occurred"
                print(f"An exception of type {type(e).__name__} occurred while parsing {model}\n")
                break


# tests for single model case

# model= "lIlBrother/ko-barTNumText"
# chatlog = contextualDataFeeding(model)
# data_extracted = metadataExtraction(chatlog)
# print(f"{model} : {data_extracted}")

# tests for single architecture model cases

# retry_count = 0
# architecture = "AlbertForPreTraining"
# for model in data[architecture]:
#     while retry_count < MAX_RETRIES:
#         try:
#             chatlog = contextualDataFeeding(model)
#             print("context done")
#             data_extracted = asyncio.run(metadataExtraction(chatlog))
#             print("extract done\n")
#             results[architecture][model]=data_extracted
#             print(f"{model} : {data_extracted}\n")
#             break
#         except openai.error.Timeout:
#             retry_count += 1
#             print(f"retry count from timeout: {retry_count}\n")
#         except openai.error.ServiceUnavailableError:
#             retry_count += 1
#             print(f"retry count from service unavilable: {retry_count}\n")
#         except ValueError:
#             results[architecture][model] = "Invalid 'model_index' in" + str(model) + ", model card could not be parsed"
#             print(f"Invalid 'model_index' in {model}, model card could not be parsed\n")
#             break
#         except huggingface_hub.utils._errors.RepositoryNotFoundError:
#             results[architecture][model] = str(model) + "repo not found"
#             print(f"{model} repo not found\n")
#             break

file_path = "result.json"
with open(file_path, "w") as json_file:
    json.dump(results, json_file, indent = 4)

end_time = time.time()
print(f"total elapsed time: {int(end_time - start_time)/3600} hours {int(end_time-start_time)/60%60} minutes {int(end_time-start_time)%60} seconds")



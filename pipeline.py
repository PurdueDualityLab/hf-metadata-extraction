from huggingface_hub import HfApi, list_models, ModelCard
import constant
import keys
import openai
import data_cleansing


hf_api = HfApi(
    endpoint="https://huggingface.co", # Can be a Private Hub endpoint.
    token= keys.HUGGINGFACE_API_KEY, # Token is not persisted on the machine.
)

# card = ModelCard.load('bert-base-uncased')
card = ModelCard.load('gpt2')
#card = ModelCard.load('runwayml/stable-diffusion-v1-5')


#card.content = card.data + card.text
print(card.content + "\n\n") 

content = data_cleansing.remove_url(card.content) 
subsections = data_cleansing.split_to_subsections(content)


openai.api_key = keys.OPENAI_API_KEY

chatlog = []
chatlog.append({"role" : "system", "content" : constant.BACKGROUND})

for section_headers in subsections:
    chatlog.append({"role": "user", "content" : "the model" + section_headers})
    chatlog.append({"role" : "assistant", "content" : subsections[section_headers]})


chat = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages = chatlog
)

print("BACKGROUND DONE !!! \n")

def chat(chatlog):
    chat = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = chatlog
    )
    chatlog.append({"role" : "assistant", "content" : chat['choices'][0]['message']['content']})
    print("\n" + chatlog[-2]["content"] + "\n")
    print("\n" + chatlog[-1]["content"] + "\n")
    return chatlog

def find_model_id(chatlog):
    chatlog.append({"role" : "user", "content" : constant.QUESTION_MODEL_ID})
    return chat(chatlog)

def find_tags(chatlog):
    chatlog.append({"role" : "user", "content" : constant.QUESTION_TAGS})
    return chat(chatlog)

def find_datasets(chatlog):
    chatlog.append({"role" : "user", "content" : constant.QUESTION_DATASETS})
    return chat(chatlog)

def find_language(chatlog):
    chatlog.append({"role" : "user", "content" : constant.QUESTION_LANGUAGE})
    return chat(chatlog)

def find_license(chatlog):
    chatlog.append({"role" : "user", "content" : constant.QUESTION_LICENSE})
    return chat(chatlog)

def find_model_task(chatlog):
    chatlog.append({"role" : "user", "content" : constant.QUESTION_MODEL_TASK})
    return chat(chatlog)

def find_param(chatlog):
    chatlog.append({"role" : "user", "content" : constant.QUESTION_PARAMETERS})
    return chat(chatlog)

def find_eval(chatlog):
    chatlog.append({"role" : "user", "content" : constant.QUESTION_EVALUATION})
    return chat(chatlog)

chatlog = find_model_id(chatlog)
chatlog = find_tags(chatlog)
chatlog = find_datasets(chatlog)
chatlog = find_language(chatlog)
chatlog = find_license(chatlog)
chatlog = find_model_task(chatlog)
chatlog = find_param(chatlog)
chatlog = find_eval(chatlog)




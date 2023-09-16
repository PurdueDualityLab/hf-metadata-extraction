import huggingface_hub
from huggingface_hub import HfApi, list_models, ModelCard
import constant
import keys
import openai
import dataCleansing


hf_api = HfApi(
    endpoint="https://huggingface.co", # Can be a Private Hub endpoint.
    token= keys.HUGGINGFACE_API_KEY, # Token is not persisted on the machine.
)
openai.api_key = keys.OPENAI_API_KEY

#load in model card when given model name on hf
def load_card(model_name):
    try:
        card = ModelCard.load(model_name)
    except huggingface_hub.utils._errors.RepositoryNotFoundError:
        print(f"{model_name} repo not found\n")
    return card.content

#set up chat with gpt given chatlog
def chat(chatlog):
    chat = openai.ChatCompletion.create(
        model = "gpt-3.5-turbo",
        messages = chatlog
    )
    chatlog.append({"role" : "assistant", "content" : chat['choices'][0]['message']['content']})
    print("\n" + chatlog[-2]["content"] + "\n")
    print("\n" + chatlog[-1]["content"] + "\n")
    return chatlog


#for future consider not appending questions after previous quenstions
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

card = load_card("gpt2")

print(card + "\n\n") 

content = dataCleansing.remove_url(card) 
subsections = dataCleansing.split_to_subsections(content)

chatlog = []
chatlog.append({"role" : "system", "content" : constant.BACKGROUND})

for section_headers in subsections:
    chatlog.append({"role": "user", "content" : "the model" + section_headers})
    chatlog.append({"role" : "assistant", "content" : subsections[section_headers]})


print("BACKGROUND DONE !!! \n")


chatlog = find_tags(chatlog)
chatlog = find_datasets(chatlog)
chatlog = find_language(chatlog)
chatlog = find_license(chatlog)
chatlog = find_model_task(chatlog)
chatlog = find_param(chatlog)
chatlog = find_eval(chatlog)




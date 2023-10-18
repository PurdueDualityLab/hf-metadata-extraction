import huggingface_hub
import openai
from collections import Counter
from langchain.text_splitter import MarkdownHeaderTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chat_models import ChatOpenAI
from langchain.chains import create_extraction_chain
from langchain.prompts import ChatPromptTemplate
from langchain.prompts.chat import SystemMessage, HumanMessagePromptTemplate
import time
import json
import argparse
import asyncio
import sys
import os

import keys
import dataPreProcessing 
import prompt

os.environ["OPENAI_API_KEY"] = keys.OPENAI_API_KEY

with open("input.json", 'r') as json_file:
    input_models = json.load(json_file)

with open("filtered_models.json", 'r') as json_file:
    data = json.load(json_file)

with open("metaSchema.json", 'r') as json_file:
    schema = json.load(json_file)


def load_card(model_name: str) -> str:

    try:
        card = huggingface_hub.ModelCard.load(model_name)
    except huggingface_hub.utils._errors.RepositoryNotFoundError:
        print(f"{model_name} repo not found\n")
        #card.data = tags
        #card.text card
        #cald.content tags+cards
    return ("#tags\n\n" + card.content)
    
def get_type_and_task(model_name: str) -> (str,str):
    model_info = huggingface_hub.hf_api.model_info(model_name)
    tags = model_info.tags

    tasks = {
    "multimodal":["feature-extraction", "text-to-image", "image-to-text", "text-to-video", "visual-question-answering", "document-question-answering", "graph-machine-learning"],
    "computer-vision":["depth-estimation", "image-classification", "object-detection", "image-segmentation", "image-to-image", "unconditional-image-generation", "video-classification", "zero-shot-image-classification"],
    "natural-language-processing":["text-classification", "token-classification", "table-question-answering", "question-answering", "zero-shot-classification", "translation", "summarization", "conversational", "text-generation", "text2text-generation", "fill-mask", "sentence-similarity"], 
    "audio":["text-to-speech", "automatic-speech-recognition", "audio-to-audio", "audio-classification", "voice-activity-detection"],
    "tabular":["tabular-classification", "tabular-regression"],
    "reinforcement-learning":["reinforcement-learning", "robotics"]
    }

    for tag in tags:
        for key in tasks.keys():
            if tag in tasks[key]:
                return (key, tag)
    return (None, None)

def get_data_schema(model_type: str) -> dict:
    general_schema = dict(schema["general_metadata"])
    if model_type:
        task_specific_schema = dict(schema["task_specific_metadata"][model_type])
        metadata_schema = {
            "properties": {**general_schema, **task_specific_schema}
            }
        return metadata_schema
    return {"properties": general_schema}

start_time = time.time()

def pretty_print_docs(docs):
    return f"\n{'-' * 100}\n".join([f"Document {i+1}:\n\n" + d.page_content for i, d in enumerate(docs)])

result = {}

#model = input("Input model ID: ")
for model in input_models:
    card = load_card(model)
    model_type, model_task = get_type_and_task(model)
    data_schema = get_data_schema(model_type)
    headers_to_split_on = [
        ("#", "header 1"),
        ("##", "header 2"),
        ("###", "header 3")
        ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on = headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(card)
    #print(md_header_splits)

    vector_store = FAISS.from_documents(md_header_splits, OpenAIEmbeddings())
    llm = OpenAI(temperature = 0)
    compressed_docs = []
    for metadata in data_schema["properties"]:
        retriever = vector_store.as_retriever(search_kwargs = {"k": 3})
        retriever_prompt = prompt.METADATA_PROMPT[metadata]
        print(f"\nprompt: {retriever_prompt}")    
        # docs = list of doc objects
        docs = retriever.get_relevant_documents(retriever_prompt)
        #print(pretty_print_docs(docs))
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor = compressor, base_retriever = retriever)
        compressed_docs.extend(compression_retriever.get_relevant_documents(retriever_prompt))
        print(pretty_print_docs(compression_retriever.get_relevant_documents(retriever_prompt)))

    extraction_prompt = """
        Given relevant documents on huggingface {model_type} model {model}, extract the properties of one single entity mentioned in the 'information_extraction' function.
        Extraction rules: 
        - Adhere strictly to the schema.
        - If a property is not present and is not required in the function parameters, do not include it in the output.
        - If a propetry is not present and is required in the function parameters, output 'None' instead.
        """
    print(f"extraction prompt: {extraction_prompt}\n\n")

    prompt_template = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content = (extraction_prompt)
            ),
            HumanMessagePromptTemplate.from_template("{documents}"),
        ]
    )

    chatbot = ChatOpenAI(temperature = 0.1, model = "gpt-3.5-turbo")

    chain = create_extraction_chain(schema = data_schema, llm = chatbot, prompt = prompt_template)

    print(data_schema)
    extraction_result = chain.run({
        "model_type": model_type,
        "model": model,
        "documents": pretty_print_docs(compressed_docs)
        })
    print("\n\n")
    print(extraction_result)

    result[model] = extraction_result

file_path = "result.json"
with open(file_path, "w") as json_file:
    json.dump(result, json_file, indent = 4)

end_time = time.time()
print(f"total elapsed time: {int(end_time - start_time)/3600} hours {int(end_time-start_time)/60%60} minutes {int(end_time-start_time)%60} seconds")

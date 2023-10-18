BACKGROUND_SINGLE= \
    "Given the following metadata schema, and information of a model from huggingface, \
    Identify the metadata of the model according to the meta schema and output them.\n \
    Formatting Rules: \n \
    - \
    - the response should only be the name of the model\n\
    - if no model was found, return null\n \
    - the response should all be lower case\n \
    - no extra text in the response\n \
    Example of good response: \n \
    Case 1: \
    Output 1: \
    Case 2: \
    Output 2: \
    Example of bad response: \n \
    Case 1: \
    Output 1: \
    Case 2: \
    Output 2: \
    "

METADATA_PROMPT = {
   "datasets": "What are the datasets used to train this model", # eval: 
    "language": "What is the language", # eval: 
    "license": "What is the license", # eval: 
    "model_tasks": "What are this model's model task", # eval: 
    "github": "What is the github repository associated to this model", # eval: 
    "paper": "What is the research paper associated to this model", # eval: 
    "frameworks": "What frameworks (pytorch, tensorflow, jax, transformers) are used in this model", # eval: 
    "parameter_count": "What are the number of parameters (#params) this model is trained on", # eval: 
    "hyper_parameters": "What are the hyper parameters (epoch, steps, learning rate, optimizer) of this model", # eval: 
    "evaluation": "What is the evalutaion of the model. What evaluation metrics where used, and what are the results", # eval: 
    "hardware": "What hardware was used to train this model", # eval: 
    "limitation_and_bias": "What are the limitations and biases of the model", # eval: 
    "how_to_use": "provide a short paragrpah on how to use the model", # eval: 
    "input_format": "What is form of input does the model take", # eval: 
    "output_format": "What is form of output does the model return", # eval: 
    "input_preprocessing": "", # eval: 
    "input_size": "",  # eval: 
    "num_of_classes_for_classification": "", # eval: 
    "tokenLimit": "What is the token limit of this NLP model", # eval: 
    "vocabulary_size": "What is the vocabulary size of this NLP model", # eval: 
    "sample_rate": "", # eval: 
    "WER": "", # eval: 
    "agent": "", # eval: 
    "training_environment": "", # eval: 
    "SB3": "" # eval: 
}
METADATA_PROMPT = {
    "datasets": "What datasets were used to train the model (include link if present)", # eval: 
    "license": "What is the license", # eval: 
    "github": "What links to github repositories are available and what these repositories are for", # eval: 
    "paper": "What is the research paper associated to this model", # eval: 
    "upstream_model": "What is the upstream model of this model", # eval:
    "parameter_count": "What are the number of parameters (#params) this model is trained on", # eval: 
    "hyper_parameters": "What are the values of some hyper parameters (parameters that control the learning process of the model) of this model.", # eval: 
    "evaluation": "What is the evalutaion of the model. What evaluation metrics where used, and what are the results (include whole table if present)", # eval: 
    "hardware": "What hardware was used to train this model", # eval: 
    "limitation_and_bias": "What are the limitations and biases of the model", # eval: 
    "demo": "Find a form of demo for the model could be a link, code snippet or short paragraph", # eval: 
    "input_format": "What is the format of the data used as input for the model", # eval: 
    "output_format": "What is the format of the data used as output of the model", # eval: 
    "input_preprocessing": "What is the input preprocessing of this model", # eval: 
    "input_size": "What is the image input size",  # eval: 
    "num_of_classes_for_classification": "How many classes for classification does this model have", # eval:
    "trigger_word": "if the model is diffusion based, does it have any trigger word", # eval: 
    "input_token_limit": "What is the input token limit of this NLP model", # eval: 
    "vocabulary_size": "What is the vocabulary size of this NLP model", # eval: 
    "sample_rate": "", # eval: 
    "WER": "", # eval: 
    "agent": "", # eval: 
    "training_environment": "", # eval: 
    "SB3": "" # eval: 
}

EXTRACTION_PROMPT = \
    "Given metadata information of huggingface {model_type} model : {model}, extract the properties of ONE single entity mentioned in the 'information_extraction' function.\n \
    Extraction rules: \n \
    - rule 1: Adhere strictly to the schema structure in 'information_extraction'\n \
    - rule 2: If a property is not present and is required in the function parameters, output 'None' instead\n \
    - rule 3: If a property is not present and is not required in the function parameters, do not include it in the output\n \
    - rule 4: Only extract one item for 'info' in  'information_extraction' function \n \
    Extraction rules for specific metadata: \n \
    - datasets: only return dataset used to train or finetune model, not the upstream model of the model\n \
    - github: extract github link of this model (only return the url)\n \
    - paper: if a research paper was written, extract arxiv research paper link\n \
    - parameter_count: The number of parameters the model was trained on, sometimes represented in the form #params\n \
    - upstream_model: provide huggingface model ID of upstream model\n \
    - hyper_parameters: extract possible hyperparameters and their\n \
    - evaluation: extract evaluation metric/tasks and their respective evaluation results\n \
    - hardware: extract any hardware used to train the model\n \
    - limitation_and_biases: extract a short summary of limitation and biases of the model\n \
    - demo: extract any links, code snippets, or small paragraphs on how to use the model\n \
    - input_format: extract the input format for this model\n \
    - output_format: extract the output format of this model\n \
"
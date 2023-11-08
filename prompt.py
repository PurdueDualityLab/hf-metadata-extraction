METADATA_PROMPT = {
    "datasets": "What datasets, provide url if possible, were the model trained/pretrained on?" , # eval: 
    "license": "What is the license?", # eval: 
    "github": "What urls to github repositories are available and what these urls are for?", # eval: 
    "paper": "What is the research paper url associated to this model?", # eval: 
    "upstream_model": "What model was this model pretrained or downstreamed from?", # eval:
    "parameter_count": "What are the number of parameters (#params) this model is trained on, sometimes represented with \"M\", \"B\", and \"T\" as million, billion, and trillion?", # eval: 
    "hyper_parameters": "What are the values of some hyper parameters (parameters that control the learning process of the model) of this model.", # eval: 
    "evaluation": "What is the evalutaion of the model. What evaluation metrics where used, and what are the results (include whole table if possible)?", # eval: 
    "hardware": "What hardware (GPU and TPU pods) was used to train this model?", # eval: 
    "limitation_and_bias": "What are the limitations and biases of the model?", # eval: 
    "demo": "Find a form of demo for the model could be a link, code snippet or short paragraph.", # eval: 
    "input_format": "What is the format of the data used as input for the model?", # eval: 
    "output_format": "What is the format of the data used as output of the model?", # eval: 
}

SIMPLE_METADATA_EXTRACTION_PROMPT = \
    "Given information on metadata of huggingface {domain} model : {model}, extract the properties of ONE single entity mentioned in the 'information_extraction' function.\n \
    Extraction rules: \n \
    - rule 1: Adhere strictly to the schema structure and in 'information_extraction'\n \
    - rule 2: If a metadata is not present but is required in the function parameters, output empty string instead\n \
    - rule 3: If a property is not present and is not required in the function parameters, do not include it in the output\n \
    - rule 4: Only extract one item for 'info' in 'information_extraction' function \n \
    Extraction rules for metadata: \n \
    - datasets: only return dataset used to train or finetune model, not the upstream model of the model\n \
    - github: extract github link of this model (only return the github link)\n \
    - paper: extract research paper url\n \
    - parameter_count: output the number of parameters, represented with \"M\", \"B\", and \"T\" as million, billion, and trillion\n \
    - upstream_model: provide huggingface model ID of upstream model\n \
    - hardware: extract any hardware used to train the model\n \
"

COMPLEX_METADATA_EXTRACTION_PROMPT = \
    "Given information on metadata of huggingface {domain} model : {model}, extract the properties of ONE single entity mentioned in the 'information_extraction' function.\n \
    Extraction rules: \n \
    - rule 1: Adhere strictly to the schema structure in 'information_extraction'\n \
    - rule 2: If a property is not present but is required in the function parameters, output empty string instead\n \
    - rule 3: If a property is not present and is not required in the function parameters, do not include it in the output\n \
    - rule 4: Only extract one item for 'info' in 'information_extraction' function \n \
    Extraction rules for metadata: \n \
    - hyper_parameter: extract possible hyperparameters\n \
    - evaluation: extract evaluation metric/tasks and their respective evaluation results\n \
    - limitation_and_biases: extract a short summary of limitation and biases of the model\n \
    - demo: extract any links, code snippets, or small paragraphs on how to use the model\n \
    - input_format: extract the input format/requirement for this model\n \
    - output_format: extract the output format of this model\n \
"
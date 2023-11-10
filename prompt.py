PREFIX_PROMPT = \
"Given model card of Huggingface {domain} model : {model} fill in the following model schema structure in 'information_extraction' function\n\
Rules:\n \
- rule 1: Adhere strictly to the schema structure\n \
- rule 2: If you are confident a property is not present in the documents, leave as empty string in the schema\n\
"


METADATA_PROMPT = \
"\
- datasets: return dataset used to train or finetune model, not the upstream model of the model\n \
- license: return this model's license\n \
- github_repo: return github repository of this model\n \
- paper: return research paper of this model\n \
- base_model: return huggingface model ID of upstream model\n \
- parameter_count: return the number of parameters, represented with \"M\", \"B\", and \"T\" as million, billion, and trillion\n \
- hardware: extract any hardware associated with the model\n \
- carbon_emitted: extract the carbon emissions amount of the model\n \
- hyper_parameters: extract possible hyperparameters\n \
- evaluation: extract evaluation metric/tasks and their respective evaluation results or a url to evaluation results\n \
- limitation_and_biases: extract a short summary of limitation and biases of the model\n \
- demo: extract any links, code snippets, or small paragraphs on how to use the model\n \
- grant: extract the grant/sponsors of this model\n \
- input_format: extract the input format/requirement for this model\n \
- output_format: extract the output format of this model\n \
"

LANG_PROMPT = " - language: return supported languages for this model\n "
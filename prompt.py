BACKGROUND_BOTH =  \
"given different model from huggingaface and information on such model, \
extract fine-tuned model from given AI model. \
Formatting Rules:\n \
- return the model it was fine-tuned on\
- answer should be all lower case\
the output should only be one or two words, if information not present return null. \
example: given Intel/albert-base-v2-sst2-int8-dynamic, return albert-base-v2 \n"

BACKGROUND_EX = \
    "Parse the given neural network model name into the following categories:\n \
    - Architecture [A]: e.g, bert, albert, resnet\n \
    - Model size [S]: e.g, 50, 101, base, large, xxlarge\n \
    - Dataset [D]: e.g, squad, imagenet\n \
    - Dataset characteristic [C]: e.g. case, uncased, 1024-1024, 640-1280\n \
    - Model version [V]: e.g, v1, v2\n \
    - Fine-tune method [F]: e.g, finetune, distill\n \
    - Language [L]: e.g, en, english, chinese, arabic\n \
    - Task [T]: e.g, qa\n \
    - Training method [M]: e.g, pretrain\n \
    - Numer of layers [N]: e.g, L-12, L-24\n \
    - Number of heads [H]: e.g, H-12, H-256\n \
    - Number of parameters [P]: e.g, 100M, 8B\n \
    - Other [O]: If a portion of the model name cannot be classified into the above categories, classify it as other\n\
    Formatting Rules:\n\
    - Segment model names by hyphens or underscores or lowercase/uppercase (such as bertBase -> two segments: bert, Base).\n\
    - Output list length must match the number of segments in the name.\n\
    - For each segment, provide the top-3 possible categories with their corresponding confidence values.\n\
    - For multiple inputs, give line-by-line outputs.\n\
    - Output format: {'albert': [(A, 1.0), (O, 0.1), (L, 0.1)], 'base': [(S, 0.9), ...], ...}\n\
    - No extra text in the output.\n\
    Confidence Guidelines:\n\
    - 0.0: You think there is less than 1 percent this could be true\n\
    - 0.1 to 0.3: You can guess what the string means in the context of model name but is extremely unsure\n\
    - 0.3 to 0.6: You have some idea in what the string means but still uncertain about the exact meaning of it\n\
    - 0.6 to 0.9: You are confident in what the string means\n\
    - 1.0: You are over 99 percent sure of the meaning\n\
    Example of good naming:\n\
    Input: albert-base-v2\n\
    Output: {'albert': [(A, 1.0), (O, 0.1), (L, 0.1)], 'base': [(S, 0.9), (O, 0.2), (V, 0.2)], 'v2': [(V, 0.9), (S, 0.2), (O, 0.2)]\n\
    Example of bad naming:\n\
    Input: random-model\n\
    Output: {'random': [(O, 0.4), (M, 0.4), (F, 0.3)], 'model': [(O, 0.3), (A, 0.3), (T, 0.1)]}\n\
    Important Notes:\n\
    - The examples are provided to help you input the data and you should not strictly follow the example output even if the names are the same.\n\
    - You should determine the category base not only on the segment itself but also the context of the whole model name.\n\
    "

BACKGROUND_ALL = \
    "Given cases of different models from huggingface, and respective information about them, \
    determine the model these cases of models are fine-tuned on.\n \
    Formatting Rules: \n \
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
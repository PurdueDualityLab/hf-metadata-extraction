BACKGROUND = "extract metadata from given AI model, the output should only be one or two words, if information not present return null \n"

QUESTION_TAGS = "what are the tags of this model, if information not present reply null"
QUESTION_DATASETS = "what are the datasets used in this model, if information not present reply null"
QUESTION_LANGUAGE = "what is the language of the model, if information not present reply null"
QUESTION_LICENSE = "what is the license used by the model, if information not present reply null"
#QUESTION_MODEL_TASK = "what tasks does the model tackle (multimodal, computer vision, natural language processing, audio, tabular, reinforcement learning), if information not inferrable reply null"
#adding "if information not inferrable reply null", limits ability to infer and will just return null
# using synonyms like guess, educated guess don't change result
QUESTION_MODEL_TASK = "what tasks does the model tackle (multimodal, computer vision, natural language processing, audio, tabular, reinforcement learning), please reply with just the task name"
#not consistent, sometimes cannot infer
#sometimes got correct answers (3/4), sometimes says does not find evaluation result
QUESTION_PARAMETERS = "how many parameters were used to train this model, if not mentioned reply null"
#QUESTION_EVALUATION = "what is the evaluation results for this model"
#QUESTION_EVALUATION = "Print out the test results from the evaluation results"
#QUESTION_EVALUATION = "Does the model card mention evaluation result. Could you print out the test results"
#for some reason does not find evaluation results in the model card
QUESTION_EVALUATION = "Does the model card mention evaluation result. Could you print out the test results? Could you print out the test results? Could you print out the test results"




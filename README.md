## description

This python script takes the models listed in filtered_models.json file, loads their model cards into LLM (gpt 3.5 turbo), and extracts the model these models were fine tuned on. If no fine-tune model were found, it will return null. 

## runtime

It took 20 hours to finish running on my laptop. The main reason for the long runtime is due to server side issues, resulting in multiple retries. 
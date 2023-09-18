import tiktoken

def num_tokens_in_chat(chat):
    encoding= tiktoken.get_encoding("cl100k_base")
    num_token = 0
    for entry in chat:
        num_token += 4  # every entry follows <im_start>{role/name}\n{content}<im_end>\n 
        for key, value in chat.items():
            num_token += len(encoding.encode(value)) 
    return num_token
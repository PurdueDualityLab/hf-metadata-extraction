# import spacy
import re

# nlp = spacy.load("en_core_web_sm") 

def split_by_subsections(card):
    
    sections = {}
    #fix top parts, subsections vs subsubsections
    subsection_content = card.split("## ")

    for section in subsection_content:
        sections[section.split("\n")[0]] = section
    return sections

def split_by_sections(subtext: str) -> int:
    
    #search for new sections
    match = re.search(r'(#+.*?)#+', subtext, re.DOTALL)
    if match:
        return len(match.groupt(1))
    return len(subtext)

# def remove_tags(text: str) -> str:
#     return re.sub(r'---.*?---', '', text, flags = re.DOTALL)

# def remove_headers(text: str) -> str:
#     return re.sub(r'(.|\n)*?#.*?$', '', text, flags = re.MULTILINE)

# def query_sentences(card: str, keywords: list) -> dict:
    
#     #process card through nlp
#     result = ""
#     doc = nlp(card)
#     for sentence in doc.sents:
#         if any(keyword in sentence.text for keyword in keywords):
#             #text = remove_tags(sentence.text)
#             text = remove_headers(text)
#             result += "-" + text.strip() + "\n"
#     return result


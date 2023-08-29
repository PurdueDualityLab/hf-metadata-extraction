import re

def remove_url(text):
    text = re.sub(r'\S*https\S*', '', text)
    return text

def split_to_subsections(text):
    sections = {}
    #fix top parts
    subsection_content = text.split("## ")

    for section in subsection_content:
        sections[section.split("\n")[0]] = section
    return sections
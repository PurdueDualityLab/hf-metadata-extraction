def split_by_subsections(card):
    
    sections = {}
    #fix top parts, subsections vs subsubsections
    subsection_content = card.split("## ")

    for section in subsection_content:
        sections[section.split("\n")[0]] = section
    return sections

import spacy
from spacy import displacy

nlp = spacy.load("en_core_web_sm")

def get_phrases(root):
    phrases = []
    for child in root.subtree:
        if child == root:
            phrases.append(root.text)
        elif child.head == root:
            if len(list(child.subtree)) > 1:
                phrases.append(get_phrases(child))
            else:
                if child.pos_ in ("DET", "PRON", "AUX"):
                    continue
                else:
                    phrases.append(child.text)

    return " ".join(phrases)

def extract_key_phrases(question):
    doc = nlp(question)
    phrases = get_phrases(doc[:].root)

    return phrases



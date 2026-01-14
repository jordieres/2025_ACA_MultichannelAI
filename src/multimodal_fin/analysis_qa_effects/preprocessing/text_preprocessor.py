import re

class TextPreprocessor:
    def __init__(self, nlp):
        self.nlp = nlp

    def clean_text_spacy(self, text: str) -> str:
        text = re.sub(r"[^a-zA-Z0-9\s]", "", str(text).lower())
        doc = self.nlp(text)
        tokens = [t.lemma_ for t in doc if not t.is_stop and len(t) > 2 and t.is_alpha]
        return " ".join(tokens)
from collections import Counter

import fasttext

class LanguageDetector:
    def __init__(self, path="../model.bin"):
        self.model = fasttext.load_model(path)

    def predict_lang(self, text, k=10):
        text = text.replace('\n', '  ')
        langs, proba = self.model.predict(text, k=k)
        res = Counter(dict(zip([lang[9:] for lang in langs], proba)))
        if 'mdf' not in res:
            res['mdf'] = 0
        return res
    
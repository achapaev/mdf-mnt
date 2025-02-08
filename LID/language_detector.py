from collections import Counter

import fasttext


class LanguageDetector:
    """
    A language detection class using a FastText model.
    """
    def __init__(self, path: str = "model.bin"):
        """
        Initializes the LanguageDetector with a FastText model.
        
        Args:
            path (str, optional): Path to the FastText model file. Default is "model.bin".
        """
        self._model = fasttext.load_model(path)

    def predict_lang(self, text: str, k: int = 10) -> Counter:
        """
        Predicts the language probabilities of a given text.
        
        Args:
            text (str): The input text for language prediction.
            k (int, optional): The number of top language predictions to return. Default is 10.
        
        Returns:
            Counter: A Counter dictionary with language codes as keys and probabilities as values.
        """
        text = text.replace('\n', '  ')
        langs, proba = self._model.predict(text, k=k)
        
        res = Counter({lang.replace("__label__", ""): prob for lang, prob in zip(langs, proba)})
        if 'mdf' not in res:
            res['mdf'] = 0
        return res

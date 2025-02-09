from itertools import groupby
import re

import razdel


def remove_hyphenation(text: str) -> str:
    """
    Removes hyphenation from a given text by merging words split with hyphens or spaces.
    
    Example:
        "по-\ нимаемый иска- женный при- мер" -> "понимаемый искаженный пример"
    
    Args:
        text (str): The input text containing hyphenated words.
    
    Returns:
        str: The text with hyphenation removed.
    """
    return re.sub(
        r'(\w)([\-+]\s+)(\w)', 
        lambda matchobj: matchobj.group(1) + matchobj.group(3), 
        text
    )


def limit_repeated_chars(text: str, max_run: int = 3) -> str:
    """
    Limits consecutive repeated characters to a specified maximum number.
    
    Example:
        "[8_________________________ 2400 3 сядт, 4 дес. 6 един." -> "[8___ 2400 3 сядт, 4 дес. 6 един."
    
    Args:
        text (str): The input text containing repeated characters.
        max_run (int, optional): The maximum number of consecutive identical characters allowed. Default is 3.
    
    Returns:
        str: The text with excessive repeated characters trimmed.
    """
    return ''.join(''.join(list(group)[:max_run]) for _, group in groupby(text))


def clean_text(raw_text: str) -> str:
    """
    Cleans the input text by performing the following operations:
    - Removing hyphenation.
    - Limiting repeated characters.
    - Replacing multiple spaces with a single space.
    - Removing asterisks at the beginning of words.
    - Normalizing spacing around periods.
    
    Args:
        raw_text (str): The input raw text.
    
    Returns:
        str: The cleaned text.
    """
    text = remove_hyphenation(raw_text)
    text = limit_repeated_chars(text)
    text = re.sub('(\. )+', '. ', text)
    text = text.replace('\xa0', ' ')
    text = re.sub('\s+', ' ', text)
    text = text.replace('* ', '')
    return text.strip()


def split_into_sentences(text: str) -> list[str]:
    """
    Splits a given text into sentences using the Razdel library.
    
    Args:
        text (str): The input text to be split.
    
    Returns:
        list[str]: A list of sentences extracted from the text.
    """
    sents = []
    for sent in razdel.sentenize(text):
        sent_text = sent.text.replace('-\n', '').replace('\n', ' ').strip()
        sents.append(sent_text)
    return sents


def is_text_valid(text: str) -> bool:
    """
    Checks if the given text meets validity criteria:
    - Contains at least one word with two or more characters.
    - Contains at least one Cyrillic letter.
    - Has a length between 3 and 500 characters.
    
    Args:
        text (str): The input text to validate.
    
    Returns:
        bool: True if the text is valid, False otherwise.
    """
    if max(len(w) for w in text.split()) < 2:
        return False
    
    if not re.match('.*[а-яё].*', text.lower()):
        return False
    
    if len(text) < 3:
        return False
    
    if len(text) > 500:
        return False
    
    return True

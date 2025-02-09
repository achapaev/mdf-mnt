import numpy as np
import torch
from transformers import BertTokenizerFast, BertModel

from utils import clean_text, split_into_sentences


def align_sentences(
        lang1_text: str, 
        lang2_text: str, 
        model: BertModel,
        tokenizer: BertTokenizerFast,
        alpha: float = 0.2,
        penalty: float = 0.2,
        threshold: float = 0.45,
    ) -> list[list[str]]:
    """
    Aligns sentences between two languages using sentence embeddings and similarity metrics.
    
    Args:
        lang1_text (str): The text in the first language.
        lang2_text (str): The text in the second language.
        model (BertModel): The embedding model.
        tokenizer (BertTokenizerFast): The tokenizer for the model.
        alpha (float, optional): The alpha parameter for similarity adjustment. Default is 0.2.
        penalty (float, optional): The penalty for alignment. Default is 0.2.
        threshold (float, optional): The similarity threshold for alignment. Default is 0.45.
    
    Returns:
        list[list[str]]: A list of aligned sentence pairs.
    """
    cleaned_lang1_text = clean_text(lang1_text)
    cleaned_lang2_text = clean_text(lang2_text)

    sents_lang1 = split_into_sentences(cleaned_lang1_text)
    sents_lang2 = split_into_sentences(cleaned_lang2_text)

    if not sents_lang1 and not sents_lang2:
        return []

    emb_lang1 = np.stack([get_sentence_embedding(s, model, tokenizer) for s in sents_lang1])
    emb_lang2 = np.stack([get_sentence_embedding(s, model, tokenizer) for s in sents_lang2])

    length_ratio = np.array([[min(len(x), len(y)) / max(len(x), len(y)) for x in sents_lang2] for y in sents_lang1])
    sims = np.maximum(0, np.dot(emb_lang1, emb_lang2.T)) ** 1 * length_ratio

    sims_rel = (sims.T - compute_topk_mean(sims) * alpha).T - compute_topk_mean(sims.T) * alpha - penalty

    alignment = compute_alignment_path(sims_rel)

    aligned_pairs = []
    for i, j in alignment:
        if sims[i, j] >= threshold:
            aligned_pairs.append([sents_lang1[i], sents_lang2[j]])

    return aligned_pairs


def get_sentence_embedding(text: str, model: BertModel, tokenizer: BertTokenizerFast, max_length: int = 128) -> np.ndarray:
    """
    Computes the sentence embedding using a transformer model.
    
    Args:
        text (str): The input text.
        model (BertModel): The embedding model.
        tokenizer (BertTokenizerFast): The tokenizer for the model.
        max_length (int, optional): The maximum token length. Default is 128.
    
    Returns:
        np.ndarray: The sentence embedding vector.
    """
    encoded_input = tokenizer(
        text, 
        padding=True, 
        truncation=True, 
        max_length=max_length, 
        return_tensors='pt'
    ).to(model.device)

    with torch.inference_mode():
        model_output = model(**encoded_input)
    embeddings = torch.nn.functional.normalize(model_output.pooler_output)
    
    return embeddings[0].detach().cpu().numpy()


def compute_topk_mean(x: np.ndarray, k: int = 5) -> np.ndarray:
    """
    Computes the mean of the top-k values in each row of a matrix.
    
    Args:
        x (np.ndarray): The input matrix.
        k (int, optional): The number of top elements to consider. Default is 5.
    
    Returns:
        np.ndarray: The mean values for each row.
    """
    m, n = x.shape
    k = min(k, n)
    topk_indices = np.argpartition(x, -k, axis=1)[:, -k:]
    rows, _ = np.indices((m, k))
    return x[rows, topk_indices].mean(1)


def compute_alignment_path(sims: np.ndarray) -> list[list[int]]:
    """
    Computes the optimal alignment path between two sets of sentences based on similarity scores.
    
    Args:
        sims (np.ndarray): The similarity matrix.
    
    Returns:
        list[list[int]]: A list of aligned index pairs.
    """
    rewards = np.zeros_like(sims)
    choices = np.zeros_like(sims, dtype=int)

    for i in range(sims.shape[0]):
        for j in range(0, sims.shape[1]):
            score_add = sims[i, j]
            if i > 0 and j > 0:
                score_add += rewards[i-1, j-1]
                choices[i, j] = 1
            best = score_add
            if i > 0 and rewards[i-1, j] > best:
                best = rewards[i-1, j]
                choices[i, j] = 2
            if j > 0 and rewards[i, j-1] > best:
                best = rewards[i, j-1]
                choices[i, j] = 3
            rewards[i, j] = best

    alignment = []
    i = sims.shape[0] - 1
    j = sims.shape[1] - 1
    while i > 0 and j > 0:
        if choices[i, j] == 1:
            alignment.append([i, j])
            i -= 1
            j -= 1
        elif choices[i, j] == 2:
            i -= 1
        else:
            j -= 1
    
    alignment.reverse()
    return alignment

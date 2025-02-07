import re

import numpy as np
import razdel
import torch


def align_ru_mdf(
        ru_text, 
        mdf_text, 
        model,
        tokenizer,
        alpha=0.2,
        penalty=0.2,
        threshold=0.45,
    ):
  
    sents_ru = process_and_sentenize(ru_text)
    sents_mdf = process_and_sentenize(mdf_text)

    emb_ru = np.stack([embed(s, model, tokenizer) for s in sents_ru])
    emb_mdf = np.stack([embed(s, model, tokenizer) for s in sents_mdf])

    pen = np.array([[min(len(x), len(y)) / max(len(x), len(y)) for x in sents_mdf] for y in sents_ru])
    sims = np.maximum(0, np.dot(emb_ru, emb_mdf.T)) ** 1 * pen

    sims_rel = (sims.T - get_top_mean_by_row(sims) * alpha).T - get_top_mean_by_row(sims.T) * alpha - penalty

    alignment = align(sims_rel)

    aligned_pairs = []
    for i, j in alignment:
        if sims[i, j] >= threshold:
            aligned_pairs.append([sents_mdf[j], sents_ru[i]])

    return aligned_pairs


def embed(text, model, tokenizer, max_length=128):
    encoded_input = tokenizer(text, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
    with torch.inference_mode():
        model_output = model.bert(**encoded_input.to(model.device))

    embeddings = torch.nn.functional.normalize(model_output.pooler_output)
    return embeddings[0].cpu().numpy()


def get_top_mean_by_row(x, k=5):
    m, n = x.shape
    k = min(k, n)
    topk_indices = np.argpartition(x, -k, axis=1)[:, -k:]
    rows, _ = np.indices((m, k))
    return x[rows, topk_indices].mean(1)


def align(sims):
    rewards = np.zeros_like(sims)
    choices = np.zeros_like(sims).astype(int)

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
    return alignment[::-1]


def process_and_sentenize(texts):
    all_sents = []

    for raw_text in texts:
        raw_text = raw_text.replace('\xa0', ' ')
        raw_text = re.sub('\s+', ' ', raw_text).strip().replace('* ', '')

        sents = []
        for sent in list(razdel.sentenize(raw_text)):
            text = sent.text.replace('-\n', '').replace('\n', ' ').strip()
            sents.append(text)
        all_sents.extend(sents)
    return all_sents

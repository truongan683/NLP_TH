from collections import Counter
from numbers import Integral
import numpy as np


def build_vocabulary(sentences, min_count=1):
    if not isinstance(min_count, Integral) or min_count < 1:
        raise ValueError("min_count must be a positive integer")
    counts = Counter(word for sentence in sentences for word in sentence)
    return {word: i for i, word in enumerate(sorted(
        word for word, count in counts.items() if count >= min_count))}


def build_cooccurrence_matrix(sentences, vocabulary, window=1):
    if not isinstance(window, Integral) or window < 1:
        raise ValueError("window must be a positive integer")
    if set(vocabulary.values()) != set(range(len(vocabulary))):
        raise ValueError("vocabulary indices must be contiguous from zero")
    matrix = np.zeros((len(vocabulary), len(vocabulary)), dtype=np.int64)
    for sentence in sentences:
        for i, word in enumerate(sentence):
            if word not in vocabulary:
                continue
            for j in range(max(0, i-window), min(len(sentence), i+window+1)):
                if j != i and sentence[j] in vocabulary:
                    matrix[vocabulary[word], vocabulary[sentence[j]]] += 1
    return matrix


def cosine_similarity(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("x and y must be vectors of equal length")
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.clip(np.dot(x, y) / denominator, -1, 1)) if denominator else 0.0


def most_similar(word, matrix, vocabulary, top_k=5):
    if word not in vocabulary:
        raise KeyError(f"Word outside vocabulary: {word}")
    if not isinstance(top_k, Integral) or top_k < 0:
        raise ValueError("top_k must be a non-negative integer")
    if np.shape(matrix) != (len(vocabulary), len(vocabulary)):
        raise ValueError("matrix must have one row and column per word")
    query = matrix[vocabulary[word]]
    scores = [(other, cosine_similarity(query, matrix[index]))
              for other, index in vocabulary.items() if other != word]
    return sorted(scores, key=lambda item: (-item[1], item[0]))[:top_k]

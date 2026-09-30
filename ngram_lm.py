

from collections import Counter


def _tokens(sentence):
    return sentence.split() if isinstance(sentence, str) else list(sentence)


def build_vocabulary(corpus):
    return sorted({word for sentence in corpus for word in _tokens(sentence)} | {"</s>"})


def count_ngrams(corpus, n):
    counts = Counter()
    for sentence in corpus:
        words = ["<s>"] * (n - 1) + _tokens(sentence) + ["</s>"]
        counts.update(tuple(words[i:i + n]) for i in range(len(words) - n + 1))
    return counts


class NGramLanguageModel:
    def __init__(self, n):
        self.n = n
        self.vocabulary = []
        self.counts = Counter()
        self.context_counts = Counter()
        self._fitted = False

    def fit(self, corpus):
        sentences = [_tokens(sentence) for sentence in corpus]
        self.vocabulary = build_vocabulary(sentences)
        self.counts = count_ngrams(sentences, self.n)
        self.context_counts = Counter()
        for ngram, count in self.counts.items():
            self.context_counts[ngram[:-1]] += count
        self._fitted = True
        return self

    def _context(self, context):
        words = _tokens(context)
        if self.n == 1:
            return ()
        return tuple((["<s>"] * (self.n - 1) + words)[-(self.n - 1):])

    def probability(self, context, word):
        key = self._context(context)
        total = self.context_counts[key]
        return self.counts[key + (word,)] / total if total else 0.0

    def sentence_probability(self, sentence):
        words = _tokens(sentence)
        result = 1.0
        for i, word in enumerate(words + ["</s>"]):
            result *= self.probability(words[:i], word)
        return result

    def next_word_distribution(self, context):
        return {word: self.probability(context, word) for word in self.vocabulary}


def train_unigram(corpus):
    return NGramLanguageModel(1).fit(corpus)


def train_bigram(corpus):
    return NGramLanguageModel(2).fit(corpus)


def train_trigram(corpus):
    return NGramLanguageModel(3).fit(corpus)


def probability(model, context, word):
    return model.probability(context, word)


def sentence_probability(model, sentence):
    return model.sentence_probability(sentence)

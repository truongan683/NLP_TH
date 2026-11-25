"""Extractive QA with explicit offset alignment, span loss and decoding."""
from collections import Counter
from dataclasses import dataclass
import re
import string

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer

BACKBONE = "google/bert_uncased_L-2_H-128_A-2"


@dataclass
class Example:
    id: str
    question: str
    context: str
    answers: tuple[tuple[str, int], ...] = ()


@dataclass
class Feature:
    example_index: int
    inputs: dict
    offsets: list
    start: int
    end: int


@dataclass
class Span:
    start: int = 0
    end: int = 0
    score: float = float("-inf")

    def text(self, context):
        return context[self.start:self.end]


def align_span(offsets, answer_start, answer_end):
    """Map a half-open character interval to inclusive context token indices."""
    start = end = None
    for index, offset in enumerate(offsets):
        if offset is None:
            continue
        left, right = offset
        if left <= answer_start < right:
            start = index
        if left < answer_end <= right:
            end = index
    if start is None or end is None or start > end or answer_start >= answer_end:
        return None
    return start, end


def make_features(examples, tokenizer, max_length, stride):
    if not examples:
        return []
    questions = [example.question.strip() for example in examples]
    # Reserve room for one context token beyond the overlap so every window advances.
    budget = max_length - stride - tokenizer.num_special_tokens_to_add(pair=True) - 1
    if stride < 0 or budget < 1:
        raise ValueError("max_length must accommodate the question, special tokens and stride")
    question_ids = tokenizer(questions, add_special_tokens=False)["input_ids"]
    if any(len(ids) > budget for ids in question_ids):
        raise ValueError("A question is too long for this max_length and stride")
    encoded = tokenizer(
        questions, [example.context for example in examples],
        truncation="only_second", max_length=max_length, stride=stride,
        return_overflowing_tokens=True, return_offsets_mapping=True,
    )
    features = []
    for window, source in enumerate(encoded["overflow_to_sample_mapping"]):
        offsets = [
            offset if segment == 1 and offset[0] < offset[1] else None
            for offset, segment in zip(encoded["offset_mapping"][window], encoded.sequence_ids(window))
        ]
        inputs = {name: encoded[name][window] for name in tokenizer.model_input_names if name in encoded}
        cls = inputs["input_ids"].index(tokenizer.cls_token_id)
        label = (cls, cls)
        for answer, char_start in examples[source].answers:
            match = align_span(offsets, char_start, char_start + len(answer))
            if match is not None:
                label = match
                break
        features.append(Feature(source, inputs, offsets, *label))
    return features


class QAModel(nn.Module):
    def __init__(self, backbone=BACKBONE):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(backbone)
        self.start_head = nn.Linear(self.encoder.config.hidden_size, 1)
        self.end_head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, **inputs):
        hidden = self.encoder(**inputs).last_hidden_state
        return self.start_head(hidden).squeeze(-1), self.end_head(hidden).squeeze(-1)


def span_loss(start_logits, end_logits, starts, ends, attention_mask):
    # Exclude padding from the probability denominator.
    padding = attention_mask == 0
    criterion = nn.functional.cross_entropy
    return (
        criterion(start_logits.masked_fill(padding, -torch.inf), starts)
        + criterion(end_logits.masked_fill(padding, -torch.inf), ends)
    ) / 2


def make_loader(features, tokenizer, batch_size, shuffle=False, seed=42):
    def collate(batch):
        inputs = tokenizer.pad([feature.inputs for feature in batch], return_tensors="pt")
        labels = tuple(torch.tensor([getattr(feature, side) for feature in batch]) for side in ("start", "end"))
        return inputs, labels, batch

    return DataLoader(
        features, batch_size=batch_size, shuffle=shuffle, collate_fn=collate,
        generator=torch.Generator().manual_seed(seed),
    )


def best_span(offsets, start_logits, end_logits, max_answer_tokens=30):
    valid = np.array([index for index, offset in enumerate(offsets) if offset is not None])
    if not valid.size:
        return Span()
    if max_answer_tokens < 1:
        raise ValueError("max_answer_tokens must be positive")
    # Score every valid start and bounded end without searching special tokens.
    end_positions = np.arange(len(valid))[:, None] + np.arange(max_answer_tokens)
    available = end_positions < len(valid)
    ends = valid[np.minimum(end_positions, len(valid) - 1)]
    scores = start_logits[valid, None] + end_logits[ends]
    row, col = np.unravel_index(np.argmax(np.where(available, scores, -np.inf)), scores.shape)
    return Span(offsets[valid[row]][0], offsets[ends[row, col]][1], float(scores[row, col]))


@torch.inference_mode()
def evaluate(model, loader, example_count, device):
    model.eval()
    predictions = [Span() for _ in range(example_count)]
    loss_sum = feature_count = 0
    for inputs, labels, features in loader:
        inputs = inputs.to(device)
        start, end = model(**inputs)
        loss = span_loss(start, end, *(label.to(device) for label in labels), inputs["attention_mask"])
        loss_sum += loss.item() * len(features)
        feature_count += len(features)
        for feature, start_scores, end_scores in zip(features, start.cpu().numpy(), end.cpu().numpy()):
            candidate = best_span(feature.offsets, start_scores, end_scores)
            index = feature.example_index
            if candidate.score > predictions[index].score:
                predictions[index] = candidate
    return loss_sum / max(feature_count, 1), predictions


def normalize(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def answer_scores(prediction, gold):
    prediction, gold = normalize(prediction), normalize(gold)
    predicted, expected = prediction.split(), gold.split()
    common = sum((Counter(predicted) & Counter(expected)).values())
    f1 = 2 * common / (len(predicted) + len(expected)) if common else 0.0
    return int(prediction == gold), f1


def score_predictions(examples, spans):
    if len(examples) != len(spans):
        raise ValueError("Expected exactly one decoded span per example")
    rows = []
    for example, span in zip(examples, spans):
        prediction = span.text(example.context)
        scores = [answer_scores(prediction, gold) for gold, _ in example.answers]
        rows.append((max(em for em, _ in scores), max(f1 for _, f1 in scores)))
    return np.array(rows, dtype=float).reshape(-1, 2)


def load_tokenizer(backbone=BACKBONE):
    tokenizer = AutoTokenizer.from_pretrained(backbone, use_fast=True, padding_side="right")
    if not tokenizer.is_fast or tokenizer.cls_token_id is None:
        raise ValueError("A fast tokenizer with a CLS token is required")
    return tokenizer

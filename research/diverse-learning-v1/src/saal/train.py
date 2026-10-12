"""REAL but small, offline-trained softmax intent model. Not LoRA or beneficial-trait RL."""
import json
import math
import random
import re
from pathlib import Path
from .beliefs import GOALS
from .canonical import digest

TRAIN_UTTERANCES = {
    "afternoon": ("I prefer later appointments", "Late in the day works", "Only evenings work", "I need afternoon times", "Mornings are difficult"),
    "few_transfers": ("I want fewer transfers", "Changing buses is hard", "Avoid multiple vehicle changes", "One direct ride matters", "Transfers exhaust me"),
    "budget": ("I need a low cost", "Keep it affordable", "Money is tight", "Choose the cheapest route", "Budget matters most"),
    "short_travel": ("I need a shorter journey", "Travel time matters", "A quick trip is best", "Get there fast", "I cannot travel far"),
}
OOD_UTTERANCES = {
    "afternoon": ("After lunch is ideal", "Not at dawn please"),
    "few_transfers": ("Please reduce connections", "Prefer an uninterrupted ride"),
    "budget": ("Minimize the fare", "I am watching expenses"),
    "short_travel": ("Avoid long commutes", "I need to arrive promptly"),
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def dataset(seed: int = 1201, per_class: int = 100, split: str = "train") -> list[tuple[str, str]]:
    if split not in {"train", "ood"}:
        raise ValueError("split")
    rng = random.Random(seed)
    templates = TRAIN_UTTERANCES if split == "train" else OOD_UTTERANCES
    samples = []
    noise = ["Please help", "I am booking this week", "Could you assist", "Thank you"]
    for goal in GOALS:
        for _ in range(per_class):
            phrase = rng.choice(templates[goal])
            if rng.random() < 0.45:
                phrase += ". " + rng.choice(noise)
            samples.append((phrase, goal))
    rng.shuffle(samples)
    return samples


class IntentClassifier:
    def __init__(self, vocab: list[str], weights: list[list[float]],
                 metadata: dict | None = None):
        if (not isinstance(vocab, list) or not vocab or
                len(vocab) != len(set(vocab)) or
                not all(isinstance(w, str) and re.fullmatch(r"[a-z]+", w) for w in vocab)):
            raise ValueError("invalid vocabulary")
        if (not isinstance(weights, list) or len(weights) != len(GOALS) or
                any(not isinstance(r, list) or len(r) != len(vocab)+1 for r in weights) or
                any(type(x) not in (int, float) or not math.isfinite(x)
                    for row in weights for x in row)):
            raise ValueError("invalid/non-finite model weights")
        if not isinstance(metadata or {}, dict):
            raise ValueError("invalid model metadata")
        self.vocab = vocab
        self.weights = weights
        self.metadata = metadata or {}
        self.index = {w: i for i, w in enumerate(vocab)}

    def predict(self, text: str) -> dict[str, float]:
        v = [0.0] * (len(self.vocab) + 1)
        v[-1] = 1.0
        for term in set(tokenize(text)):
            if term in self.index:
                v[self.index[term]] = 1.0
        scores = [sum(x*y for x, y in zip(row, v)) for row in self.weights]
        center = max(scores)
        exps = [math.exp(s-center) for s in scores]
        scale = sum(exps)
        return {g: exps[j]/scale for j, g in enumerate(GOALS)}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"model_type": "linear_intent_softmax_v1",
                     "vocab": self.vocab, "weights": self.weights,
                     "metadata": self.metadata}, sort_keys=True, indent=2) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "IntentClassifier":
        data = json.loads(Path(path).read_text())
        if data.get("model_type") != "linear_intent_softmax_v1":
            raise ValueError("unsupported model")
        return cls(data["vocab"], data["weights"], data["metadata"])


def fit_intent_model(seed: int = 1201, epochs: int = 55, per_class: int = 100) -> IntentClassifier:
    rng = random.Random(seed)
    samples = dataset(seed=seed, per_class=per_class, split="train")
    vocab = sorted({t for text, _ in samples for t in tokenize(text)})
    model = IntentClassifier(vocab, [[0.0] * (len(vocab)+1) for _ in GOALS],
          {"seed": seed, "epochs": epochs, "train_split": "train",
           "dataset_digest": digest([[t,g] for t,g in samples]), "trained_offline": True})
    for epoch in range(epochs):
        rng.shuffle(samples)
        rate = 0.17 / (1.0 + 0.045*epoch)
        for text, goal in samples:
            used = sorted(set(tokenize(text)) & model.index.keys())
            idx = [model.index[w] for w in used] + [len(vocab)]
            ps = model.predict(text)
            for j, label in enumerate(GOALS):
                delta = rate * (float(label == goal) - ps[label])
                for k in idx:
                    model.weights[j][k] += delta
    return model

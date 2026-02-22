import os
import statistics

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = "artifacts/va_prediction"
TOKENIZER_DIR = "artifacts/va_prediction"

_tokenizer = None
_model = None
_label_names = None


def _sorted_label_keys(keys) -> list:
    try:
        return sorted(keys, key=lambda k: int(k))
    except Exception:
        return sorted(keys)


def _load_label_map_txt() -> list[str] | None:
    path = os.path.join(MODEL_DIR, "label_map.txt")
    if not os.path.exists(path):
        return None
    labels: dict[int, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if not s or "=" not in s:
                    continue
                left, right = s.split("=", 1)
                idx = int(left.strip())
                name = right.strip()
                labels[idx] = name
    except Exception:
        return None
    if not labels:
        return None
    return [labels[i] for i in sorted(labels)]


def load_model():
    global _tokenizer, _model, _label_names
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, local_files_only=True
        )
        _model.eval()

        cfg = getattr(_model, "config", None)
        labels_from_file = _load_label_map_txt()
        if labels_from_file:
            _label_names = labels_from_file
        else:
            id2label = getattr(cfg, "id2label", None) or {}
            if isinstance(id2label, dict) and id2label:
                try:
                    _label_names = [str(id2label[i]) for i in _sorted_label_keys(id2label.keys())]
                except Exception:
                    _label_names = None

        if not _label_names:
            num_labels = int(getattr(getattr(_model, "config", None), "num_labels", 0) or 0)
            _label_names = [f"LABEL_{i}" for i in range(num_labels)] if num_labels else ["valence", "arousal"]

        if cfg is not None:
            if _label_names:
                if not getattr(cfg, "num_labels", None) or cfg.num_labels != len(_label_names):
                    cfg.num_labels = len(_label_names)
                cfg.id2label = {i: name for i, name in enumerate(_label_names)}
                cfg.label2id = {name: i for i, name in enumerate(_label_names)}

    return _tokenizer, _model, _label_names


def _quadrant(valence: float, arousal: float, threshold: float = 0.0) -> str:
    if valence >= threshold and arousal >= threshold:
        return "HVHA"
    if valence >= threshold and arousal < threshold:
        return "HVLA"
    if valence < threshold and arousal >= threshold:
        return "LVHA"
    return "LVLA"


def predict_va_single(text: str, *, max_length: int = 256) -> dict[str, float | str]:
    tokenizer, model, label_names = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    with torch.no_grad():
        logits = model(**inputs).logits.squeeze(0)

    values = [float(v) for v in logits.tolist()]
    raw = {label_names[i]: values[i] for i in range(min(len(label_names), len(values)))}
    valence = float(raw.get("valence", values[0] if values else 0.0))
    arousal = float(raw.get("arousal", values[1] if len(values) > 1 else 0.0))
    return {
        "valence": valence,
        "arousal": arousal,
        "quadrant": _quadrant(valence, arousal),
    }


def summarize_va(per_review: list[dict]) -> dict:
    cleaned = [x for x in (per_review or []) if isinstance(x, dict)]
    if not cleaned:
        return {
            "total": 0,
            "mean_valence": 0.0,
            "mean_arousal": 0.0,
            "std_valence": 0.0,
            "std_arousal": 0.0,
            "quadrants": {},
        }

    valences = [float(x.get("valence", 0.0)) for x in cleaned]
    arousals = [float(x.get("arousal", 0.0)) for x in cleaned]
    quadrants: dict[str, int] = {}
    for x in cleaned:
        q = str(x.get("quadrant", ""))
        quadrants[q] = quadrants.get(q, 0) + 1

    total = len(cleaned)
    quad_percentages = {
        k: (v / total) * 100.0 for k, v in sorted(quadrants.items(), key=lambda kv: -kv[1])
    }

    return {
        "total": total,
        "mean_valence": float(statistics.fmean(valences)),
        "mean_arousal": float(statistics.fmean(arousals)),
        "std_valence": float(statistics.pstdev(valences)) if total > 1 else 0.0,
        "std_arousal": float(statistics.pstdev(arousals)) if total > 1 else 0.0,
        "quadrants": quadrants,
        "quadrant_percentages": quad_percentages,
    }

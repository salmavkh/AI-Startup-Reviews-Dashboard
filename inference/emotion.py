import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Local emotion model artifacts
MODEL_DIR = "artifacts/distilbert_emotweet28"
TOKENIZER_DIR = "artifacts/distilbert_emotweet28"

_tokenizer = None
_model = None
_label_names = None


def _load_emotions_txt() -> list[str] | None:
    path = os.path.join(MODEL_DIR, "emotions.txt")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines or None
    except Exception:
        return None


def _sorted_label_keys(keys) -> list:
    try:
        return sorted(keys, key=lambda k: int(k))
    except Exception:
        return sorted(keys)


def load_model():
    global _tokenizer, _model, _label_names
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, local_files_only=True
        )
        _model.eval()

        # Prefer emotions.txt if present (explicit label order)
        cfg = getattr(_model, "config", None)
        emotions_txt = _load_emotions_txt()
        if emotions_txt:
            _label_names = emotions_txt
        else:
            # Fallback to id2label from config if available
            id2label = getattr(cfg, "id2label", None) or {}
            if isinstance(id2label, dict) and id2label:
                try:
                    _label_names = [str(id2label[i]) for i in _sorted_label_keys(id2label.keys())]
                except Exception:
                    _label_names = None

        if not _label_names:
            # Fallback: generic labels
            num_labels = int(getattr(getattr(_model, "config", None), "num_labels", 0) or 0)
            _label_names = [f"LABEL_{i}" for i in range(num_labels)] if num_labels else None

        # Ensure num_labels is set if missing
        if cfg is not None:
            if _label_names:
                if not getattr(cfg, "num_labels", None) or cfg.num_labels != len(_label_names):
                    cfg.num_labels = len(_label_names)
                # Ensure label mapping aligns with loaded labels
                cfg.id2label = {i: name for i, name in enumerate(_label_names)}
                cfg.label2id = {name: i for i, name in enumerate(_label_names)}

    return _tokenizer, _model, _label_names


def predict_proba_single(text: str, *, max_length: int = 256) -> dict[str, float]:
    """Return per-emotion probabilities for a single text."""
    tokenizer, model, label_names = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=max_length,
    )

    with torch.no_grad():
        logits = model(**inputs).logits  # [1, num_labels]

        if os.getenv("EMOTION_DEBUG"):
            lmin = float(logits.min())
            lmax = float(logits.max())
            lmean = float(logits.mean())
            print(f"[emotion] logits min={lmin:.4f} max={lmax:.4f} mean={lmean:.4f}")

        problem_type = getattr(getattr(model, "config", None), "problem_type", None)
        if problem_type == "regression":
            # Regression: logits are emotion intensities (optionally clamp to [0,1])
            probs = logits.squeeze(0).clamp(0.0, 1.0)
        elif problem_type == "multi_label_classification":
            # Multi-label: independent sigmoid per label
            probs = torch.sigmoid(logits).squeeze(0)
        else:
            # Single-label classification: softmax over labels
            probs = torch.softmax(logits, dim=-1).squeeze(0)

    probs_list = [float(p) for p in probs]

    if label_names and len(label_names) == len(probs_list):
        return {label_names[i]: probs_list[i] for i in range(len(probs_list))}
    return {f"LABEL_{i}": probs_list[i] for i in range(len(probs_list))}


def predict_label_single(text: str) -> tuple[str, float]:
    """Return (top_emotion_label, confidence)."""
    dist = predict_proba_single(text)
    label = max(dist, key=dist.get)
    return label, float(dist[label])


def emotion_percentages(
    texts: list[str],
    *,
    method: str = "count",
) -> dict:
    """Compute emotion percentages over a list of texts.

    method:
      - "count": percentage by argmax label per review (0..100)
      - "prob":  average emotion scores across reviews (0..1 for regression)

    Returns:
      {
        "method": str,
        "total": int,
        "percentages": {label: float},  # 0..100
        "counts": {label: int} | None,
      }
    """
    texts = [t.strip() for t in (texts or []) if t and str(t).strip()]
    if not texts:
        return {"method": method, "total": 0, "percentages": {}, "counts": {}}

    _tokenizer, _model, label_names = load_model()

    if method not in {"count", "prob"}:
        raise ValueError("method must be 'count' or 'prob'")

    if method == "count":
        counts: dict[str, int] = {}
        for t in texts:
            lbl, _conf = predict_label_single(t)
            counts[lbl] = counts.get(lbl, 0) + 1

        total = len(texts)
        percentages = {k: (v / total) * 100.0 for k, v in counts.items()}

        # stable ordering (desc)
        percentages = dict(sorted(percentages.items(), key=lambda kv: -kv[1]))
        counts = dict(sorted(counts.items(), key=lambda kv: -kv[1]))

        return {"method": method, "total": total, "percentages": percentages, "counts": counts}

    # method == "prob"
    summed: dict[str, float] = {}
    for t in texts:
        dist = predict_proba_single(t)
        for lbl, p in dist.items():
            summed[lbl] = summed.get(lbl, 0.0) + float(p)

    total = len(texts)
    avg = {lbl: v / total for lbl, v in summed.items()}
    percentages = dict(sorted(avg.items(), key=lambda kv: -kv[1]))

    return {"method": method, "total": total, "percentages": percentages, "counts": None}

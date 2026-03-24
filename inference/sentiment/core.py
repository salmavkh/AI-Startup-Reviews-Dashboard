import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "artifacts/roberta_sentiment/model"
TOKENIZER_DIR = "artifacts/roberta_sentiment/tokenizer"

# Trained config says: num_labels = 2, label_names = ["negative", "positive"]
NEG_IDX = 0
POS_IDX = 1

_tokenizer = None
_model = None


def load_model():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        _tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR, local_files_only=True)
        _model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_DIR, local_files_only=True
        )
        _model.eval()
    return _tokenizer, _model


def predict_single(text: str, *, threshold: float = 0.5):
    """
    Returns: (label_str, confidence_float)

    - label_str is "Positive" or "Negative"
    - confidence is the probability of the predicted label

    If you want an "Uncertain" outcome, raise `threshold` above 0.5
    (e.g., 0.6 / 0.7) and treat below-threshold as uncertain.
    """
    tokenizer, model = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        logits = model(**inputs).logits  # shape: [1, 2]
        probs = torch.softmax(logits, dim=-1).squeeze(0)  # shape: [2]

    idx = int(torch.argmax(probs))
    conf = float(probs[idx])

    # Optional "uncertain" gating (only if you actually want it)
    if conf < threshold:
        return "Uncertain", conf

    if idx == POS_IDX:
        return "Positive", conf
    return "Negative", conf

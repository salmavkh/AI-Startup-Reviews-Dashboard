import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = "artifacts/roberta_sentiment/model"
TOKENIZER_DIR = "artifacts/roberta_sentiment/tokenizer"

NEG_IDX = 0
NEU_IDX = 1
POS_IDX = 2
NEUTRAL_POLICY = "ignore"  # or "to_negative"

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


def predict_single(text: str):
    tokenizer, model = load_model()

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256,
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0)

    idx = int(torch.argmax(probs))
    conf = float(probs[idx])

    if idx == POS_IDX:
        return "Positive", conf
    if idx == NEG_IDX:
        return "Negative", conf

    if NEUTRAL_POLICY == "to_negative":
        return "Negative", float(probs[NEG_IDX])
    return "Uncertain", conf

"""
Local fine-tuned RoBERTa + BiLSTM classifier for mental health status detection.

Architecture (auto-detected from model.safetensors):
  - If weights contain 'lstm' keys  → RoBERTa + BiLSTM + Linear(768→5)
  - If weights contain config.json  → Standard RobertaForSequenceClassification

5-class label order (alphabetical, as trained):
  0 → anxiety
  1 → depression
  2 → normal
  3 → stress
  4 → suicidal

Model loading order:
  1. If model.safetensors is present in the same directory → use local files.
  2. Otherwise → download from the HF Hub repo set in the HF_MODEL_REPO env var.
     Set HF_TOKEN env var too if the repo is private.
"""

import os
from functools import lru_cache

import torch
import torch.nn as nn
from transformers import AutoTokenizer, RobertaConfig, RobertaModel

MODEL_DIR = os.path.dirname(os.path.abspath(__file__))

# Public 5-category list (stable order matching training label indices)
CATEGORIES = ["anxiety", "depression", "normal", "stress", "suicidal"]

# RoBERTa-base config matching the saved weights
_ROBERTA_CFG = RobertaConfig(
    vocab_size=50265,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=514,
    type_vocab_size=1,
)


class _RoBERTaBiLSTMClassifier(nn.Module):
    """RoBERTa + BiLSTM + linear head — matches model.safetensors (209 keys)."""

    def __init__(self):
        super().__init__()
        self.roberta = RobertaModel(_ROBERTA_CFG, add_pooling_layer=True)
        self.lstm = nn.LSTM(
            input_size=768,
            hidden_size=384,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(768, len(CATEGORIES))

    def forward(self, input_ids, attention_mask=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state       # (batch, seq, 768)
        _, (h_n, _) = self.lstm(sequence_output)
        h_cat = torch.cat([h_n[0], h_n[1]], dim=-1)      # (batch, 768)
        return self.classifier(h_cat)                     # (batch, 5)


def _model_source() -> str:
    """Return local path if model files exist, else the HF Hub repo ID."""
    if os.path.isfile(os.path.join(MODEL_DIR, "model.safetensors")):
        return MODEL_DIR
    hf_repo = os.environ.get("HF_MODEL_REPO")
    if hf_repo:
        return hf_repo
    raise RuntimeError(
        "Model not found. Either place model.safetensors in the app directory "
        "or set the HF_MODEL_REPO environment variable."
    )


def _has_lstm(weights_path: str) -> bool:
    """Return True if the safetensors file contains BiLSTM weights."""
    from safetensors import safe_open
    with safe_open(weights_path, framework="pt") as f:
        return any("lstm" in k for k in f.keys())


@lru_cache(maxsize=1)
def _load_model():
    """Load tokenizer + model once and cache them. Auto-detects architecture."""
    source = _model_source()
    hf_token = os.environ.get("HF_TOKEN", None)
    tokenizer = AutoTokenizer.from_pretrained(source, token=hf_token)

    local_weights = os.path.join(source, "model.safetensors")
    if os.path.isfile(local_weights):
        from safetensors.torch import load_file
        if _has_lstm(local_weights):
            # BiLSTM architecture
            model = _RoBERTaBiLSTMClassifier()
            model.load_state_dict(load_file(local_weights), strict=True)
        else:
            # Standard RobertaForSequenceClassification
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(source, token=hf_token)
    else:
        # HF Hub — download and detect
        from huggingface_hub import hf_hub_download
        weights_path = hf_hub_download(
            repo_id=source, filename="model.safetensors", token=hf_token
        )
        from safetensors.torch import load_file
        if _has_lstm(weights_path):
            model = _RoBERTaBiLSTMClassifier()
            model.load_state_dict(load_file(weights_path), strict=True)
        else:
            from transformers import AutoModelForSequenceClassification
            model = AutoModelForSequenceClassification.from_pretrained(source, token=hf_token)

    model.eval()
    return tokenizer, model


def classify(text: str) -> dict:
    """
    Run the fine-tuned RoBERTa model on *text*.

    Returns a dict with:
      - category (str): one of the 5 categories
      - confidence (int): 0-100 confidence for the top category
      - all_scores (dict): {category: int_score} for all 5 categories
      - model_used (str): "bert-finetuned"
    """
    text = text.strip()
    if not text:
        return {
            "category": "normal",
            "confidence": 0,
            "all_scores": {c: 0 for c in CATEGORIES},
            "model_used": "bert-finetuned",
        }

    try:
        tokenizer, model = _load_model()
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True,
        )
        with torch.no_grad():
            out = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
            )
            # BiLSTM returns a raw tensor; standard model returns an object with .logits
            logits = out if isinstance(out, torch.Tensor) else out.logits
            probs = torch.softmax(logits, dim=-1)[0]

        scores = {c: probs[i].item() for i, c in enumerate(CATEGORIES)}
        category = max(scores, key=scores.get)
        confidence = int(round(scores[category] * 100))
        all_scores = {c: int(round(v * 100)) for c, v in scores.items()}

        return {
            "category": category,
            "confidence": confidence,
            "all_scores": all_scores,
            "model_used": "bert-finetuned",
        }

    except Exception as exc:
        return {
            "category": "normal",
            "confidence": 0,
            "all_scores": {c: 0 for c in CATEGORIES},
            "model_used": "bert-finetuned",
            "error": str(exc),
        }

from __future__ import annotations

import json
import os
import pathlib
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, jsonify, request
from transformers import AutoModel, AutoTokenizer

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from models.custom_bilstm import BiLSTMClassifier
from models.vocab import Vocab

app = Flask(__name__)

# =========================
# Config por entorno
# =========================
BILSTM_RAND_DIR = os.getenv("BILSTM_RAND_DIR", "logic/artifacts/bilstm_rand")
BILSTM_W2V_DIR = os.getenv("BILSTM_W2V_DIR", "logic/artifacts/bilstm_w2v")

BERT_DIR = os.getenv("BERT_DIR", "logic/artifacts/bert") 
BERT_BASE_MODEL = os.getenv("BERT_BASE_MODEL", "dccuchile/bert-base-spanish-wwm-cased")

MIN_WORDS = int(os.getenv("MIN_WORDS", "30"))
CONF_BAND = float(os.getenv("CONF_BAND", "0.05"))
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "bilstm_rand")  # bilstm_rand | bilstm_w2v | bert

# =========================
# Device
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# Helpers de carga
# =========================
def load_vocab_and_config(model_dir: str) -> Tuple[Vocab, dict]:
    model_dir_path = pathlib.Path(model_dir)

    with (model_dir_path / "vocab.json").open("r", encoding="utf-8") as f:
        vocab_cfg = json.load(f)

    vocab = Vocab(
        stoi={tok: i for i, tok in enumerate(vocab_cfg["itos"])},
        itos=vocab_cfg["itos"],
        pad_index=vocab_cfg["pad_index"],
        unk_index=vocab_cfg["unk_index"],
    )

    with (model_dir_path / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    return vocab, cfg


def safe_vocab_encode(vocab: Vocab, text: str) -> list[int]:
    """
    Tu API antigua usa VOCAB.encode(text). En el Vocab que te di,
    el método se llama encode_text(). Esto permite ambas.
    """
    if hasattr(vocab, "encode"):
        return vocab.encode(text)  # type: ignore[attr-defined]
    return vocab.encode_text(text)  # type: ignore[attr-defined]


@dataclass
class BiLSTMBundle:
    name: str
    model_dir: str
    model: Optional[BiLSTMClassifier] = None
    vocab: Optional[Vocab] = None
    cfg: Optional[dict] = None
    threshold: float = 0.5
    max_len: int = 256
    loaded: bool = False
    error: Optional[str] = None


class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(1)


@dataclass
class BERTBundle:
    name: str = "bert"
    model_dir: str = BERT_DIR
    base_model: str = BERT_BASE_MODEL

    tokenizer: Optional[Any] = None
    bert: Optional[Any] = None  # transformers model
    mlp: Optional[MLPClassifier] = None

    cfg: Optional[dict] = None
    threshold: float = 0.5
    input_dim: int = 768
    hidden_dim: int = 256

    loaded: bool = False
    error: Optional[str] = None


def load_bilstm_bundle(name: str, model_dir: str) -> BiLSTMBundle:
    b = BiLSTMBundle(name=name, model_dir=model_dir)
    try:
        vocab, cfg = load_vocab_and_config(model_dir)
        b.vocab = vocab
        b.cfg = cfg
        b.threshold = float(cfg.get("threshold", 0.5))
        b.max_len = int(cfg.get("max_len", 256))

        model = BiLSTMClassifier(
            vocab_size=len(vocab.itos),
            embed_dim=int(cfg.get("embed_dim", 256)),
            hidden_dim=int(cfg.get("hidden_dim", 256)),
            num_layers=int(cfg.get("num_layers", 1)),
            pad_index=vocab.pad_index,
            dropout=0.3,
        ).to(device)

        state_dict = torch.load(pathlib.Path(model_dir) / "model.pt", map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

        b.model = model
        b.loaded = True
    except Exception as e:
        b.error = str(e)
        b.loaded = False
    return b


def load_bert_bundle(model_dir: str, base_model: str) -> BERTBundle:
    b = BERTBundle(model_dir=model_dir, base_model=base_model)
    try:
        model_dir_path = pathlib.Path(model_dir)
        with (model_dir_path / "config.json").open("r", encoding="utf-8") as f:
            cfg = json.load(f)

        b.cfg = cfg
        b.threshold = float(cfg.get("threshold", 0.5))
        b.input_dim = int(cfg.get("input_dim", 768))
        b.hidden_dim = int(cfg.get("hidden_dim", 256))

        # Load tokenizer + BERT base (para embeddings)
        b.tokenizer = AutoTokenizer.from_pretrained(base_model)
        b.bert = AutoModel.from_pretrained(base_model, use_safetensors=True).to(device)
        b.bert.eval()

        # Load MLP trained on embeddings
        b.mlp = MLPClassifier(input_dim=b.input_dim, hidden_dim=b.hidden_dim, dropout=0.3).to(device)
        state_dict = torch.load(model_dir_path / "model.pt", map_location=device)
        b.mlp.load_state_dict(state_dict)
        b.mlp.eval()

        b.loaded = True
    except Exception as e:
        b.error = str(e)
        b.loaded = False
    return b


# =========================
# Carga en arranque
# =========================
BILSTM_RAND = load_bilstm_bundle("bilstm_rand", BILSTM_RAND_DIR)
BILSTM_W2V = load_bilstm_bundle("bilstm_w2v", BILSTM_W2V_DIR)
BERT = load_bert_bundle(BERT_DIR, BERT_BASE_MODEL)

MODEL_REGISTRY: Dict[str, Any] = {
    "bilstm_rand": BILSTM_RAND,
    "bilstm_w2v": BILSTM_W2V,
    "bert": BERT,
}


# =========================
# Inferencia
# =========================
def confidence_label(prob: float, thr: float) -> str:
    return "low" if abs(prob - thr) < CONF_BAND else "high"


def predict_bilstm(bundle: BiLSTMBundle, text: str) -> Dict[str, Any]:
    assert bundle.model is not None and bundle.vocab is not None

    ids = safe_vocab_encode(bundle.vocab, text)
    if len(ids) == 0:
        ids = [bundle.vocab.unk_index]

    # truncar a max_len
    if bundle.max_len and len(ids) > bundle.max_len:
        ids = ids[: bundle.max_len]

    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    lengths = torch.tensor([len(ids)], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = bundle.model(input_ids, lengths)
        prob = torch.sigmoid(logits).item()

    decision = "IA" if prob >= bundle.threshold else "humano"
    return {
        "model": bundle.name,
        "score": float(prob),
        "decision": decision,
        "threshold": float(bundle.threshold),
        "confidence": confidence_label(prob, bundle.threshold),
        "confidence_band": float(CONF_BAND),
    }


def bert_embed_meanpool(bundle: BERTBundle, text: str, max_length: int = 256) -> np.ndarray:
    assert bundle.tokenizer is not None and bundle.bert is not None
    enc = bundle.tokenizer(
        [text],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = bundle.bert(**enc)
        hidden = out.last_hidden_state  # (1, L, H)
        mask = enc["attention_mask"].unsqueeze(-1)  # (1, L, 1)
        vec = (hidden * mask).sum(dim=1) / mask.sum(dim=1)  # (1, H)
    return vec.detach().cpu().numpy()[0]


def predict_bert(bundle: BERTBundle, text: str) -> Dict[str, Any]:
    assert bundle.mlp is not None

    x = bert_embed_meanpool(bundle, text, max_length=256)
    x_t = torch.from_numpy(x).float().unsqueeze(0).to(device)

    with torch.no_grad():
        logits = bundle.mlp(x_t)
        prob = torch.sigmoid(logits).item()

    decision = "IA" if prob >= bundle.threshold else "humano"
    return {
        "model": bundle.name,
        "score": float(prob),
        "decision": decision,
        "threshold": float(bundle.threshold),
        "confidence": confidence_label(prob, bundle.threshold),
        "confidence_band": float(CONF_BAND),
    }


# =========================
# Endpoints
# =========================
@app.get("/health")
def health():
    return jsonify(
        {
            "status": "ok",
            "device": str(device),
            "min_words": MIN_WORDS,
            "models": {
                k: {
                    "loaded": v.loaded,
                    "error": v.error,
                    "threshold": getattr(v, "threshold", None),
                    "model_dir": getattr(v, "model_dir", None),
                }
                for k, v in MODEL_REGISTRY.items()
            },
        }
    )


@app.get("/models")
def list_models():
    return jsonify({"available": list(MODEL_REGISTRY.keys()), "default": DEFAULT_MODEL})


@app.post("/predict")
def predict():
    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()
    model_name = (data.get("model") or request.args.get("model") or DEFAULT_MODEL).strip()

    if not text:
        return jsonify({"error": "texto vacío"}), 400

    num_words = len(text.split())
    if num_words < MIN_WORDS:
        return jsonify(
            {
                "model": model_name,
                "decision": "indeterminado",
                "reason": "texto demasiado corto",
                "min_words": MIN_WORDS,
                "text_length": num_words,
            }
        ), 200

    if model_name not in MODEL_REGISTRY:
        return jsonify({"error": f"modelo desconocido: {model_name}", "available": list(MODEL_REGISTRY.keys())}), 400

    bundle = MODEL_REGISTRY[model_name]
    if not bundle.loaded:
        return jsonify({"error": f"modelo {model_name} no cargado", "details": bundle.error}), 500

    if model_name in ("bilstm_rand", "bilstm_w2v"):
        out = predict_bilstm(bundle, text)
    else:
        out = predict_bert(bundle, text)

    out["text_length"] = num_words
    out["min_words"] = MIN_WORDS
    return jsonify(out)


@app.post("/predict/<model_name>")
def predict_named(model_name: str):
    data = request.get_json(force=True) or {}
    data["model"] = model_name
    # reutiliza /predict
    request.args = request.args.copy()
    return predict()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)

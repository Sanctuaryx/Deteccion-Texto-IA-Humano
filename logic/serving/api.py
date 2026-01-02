from __future__ import annotations

import json
import os
import pathlib

import torch
from flask import Flask, jsonify, request

from logic.models.custom_bilstm import BiLSTMClassifier
from logic.models.vocab import Vocab

app = Flask(__name__)

MODEL_DIR = os.getenv("MODEL_DIR", "artifacts/custom_model")
MIN_WORDS = int(os.getenv("MIN_WORDS", "30"))
CONF_BAND = float(os.getenv("CONF_BAND", "0.05"))  # banda de “zona gris”


def load_vocab_and_config(model_dir: str):
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    VOCAB, CONFIG = load_vocab_and_config(MODEL_DIR)

    model = BiLSTMClassifier(
        vocab_size=len(VOCAB.itos),
        embed_dim=CONFIG["embed_dim"],
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG.get("num_layers", 1),
        pad_index=VOCAB.pad_index,
        dropout=0.3,
    ).to(device)

    state_dict = torch.load(
        pathlib.Path(MODEL_DIR) / "model.pt",
        map_location=device,
    )
    model.load_state_dict(state_dict)
    model.eval()

    THRESHOLD = float(CONFIG.get("threshold", 0.5))
    MODEL_LOADED = True
except Exception as e:
    # Si falla la carga, lo indicamos en /health
    print(f"Error cargando modelo o vocabulario desde {MODEL_DIR}: {e}")
    model = None
    VOCAB = None
    CONFIG = {}
    THRESHOLD = float(os.getenv("THRESHOLD", "0.5"))
    MODEL_LOADED = False


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_dir": MODEL_DIR,
        "model_loaded": MODEL_LOADED,
        "threshold": THRESHOLD,
        "min_words": MIN_WORDS,
    }


@app.post("/predict")
def predict():
    if not MODEL_LOADED or model is None or VOCAB is None:
        return (
            jsonify(
                {
                    "error": "Modelo no cargado. Asegúrate de entrenar y guardar el modelo en MODEL_DIR."
                }
            ),
            500,
        )

    data = request.get_json(force=True) or {}
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "texto vacío"}), 400

    num_words = len(text.split())
    if num_words < MIN_WORDS:
        return jsonify(
            {
                "decision": "indeterminado",
                "reason": "texto demasiado corto",
                "min_words": MIN_WORDS,
                "text_length": num_words,
            }
        ), 200

    # Codificar texto a IDs con el vocab propio
    ids = VOCAB.encode(text)
    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)
    lengths = torch.tensor([len(ids)], dtype=torch.long).to(device)

    with torch.no_grad():
        logits = model(input_ids, lengths)  # (1,)
        prob = torch.sigmoid(logits).item()

    decision = "IA" if prob >= THRESHOLD else "humano"

    # Banda de confianza: zona gris alrededor del umbral
    if abs(prob - THRESHOLD) < CONF_BAND:
        confidence = "low"
    else:
        confidence = "high"

    return jsonify(
        {
            "score": float(prob),
            "decision": decision,
            "threshold": THRESHOLD,
            "confidence": confidence,
            "confidence_band": CONF_BAND,
            "text_length": num_words,
            "min_words": MIN_WORDS,
        }
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8001, debug=True)

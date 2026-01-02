# build_bert_embeddings.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_NAME = "dccuchile/bert-base-spanish-wwm-cased"

def encode_texts(texts, model, tokenizer, device, batch_size=8, max_length=256):
    all_vecs = []

    for i in tqdm(range(0, len(texts), batch_size), desc="BERT embeddings"):
        batch_texts = texts[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            out = model(**enc)
            # depende del modelo: out.last_hidden_state (B, L, H)
            hidden = out.last_hidden_state
            # media de tokens como embedding de frase:
            mask = enc["attention_mask"].unsqueeze(-1)  # (B, L, 1)
            masked_hidden = hidden * mask
            sum_hidden = masked_hidden.sum(dim=1)  # (B, H)
            lengths = mask.sum(dim=1)  # (B, 1)
            vecs = (sum_hidden / lengths).cpu().numpy()
        all_vecs.append(vecs)

    return np.vstack(all_vecs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    if "text" not in df.columns or "generated" not in df.columns:
        raise ValueError("CSV debe tener columnas 'text' y 'generated'.")

    texts = df["text"].astype(str).tolist()
    labels = df["generated"].astype(int).to_numpy()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME, use_safetensors=True).to(device)
    model.eval()

    vecs = encode_texts(texts, model, tokenizer, device)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, X=vecs, y=labels)
    print(f"Guardado {out_path} con X (embeddings) e y (labels).")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from gensim.models import Word2Vec

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.vocab import simple_tokenize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default="data/train.csv")
    parser.add_argument("--out_path", type=str, default="iatd/artifacts/bilstm_w2v")
    parser.add_argument("--vector_size", type=int, default=256)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--min_count", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    train_path = Path(args.train_path)
    df = pd.read_csv(train_path)

    if "text" not in df.columns:
        raise ValueError("El CSV debe tener columna 'text'.")

    sentences = []
    for text in df["text"].astype(str).tolist():
        tokens = simple_tokenize(text)
        sentences.append(tokens)

    print("Entrenando Word2Vec...")
    w2v = Word2Vec(
        sentences=sentences,
        vector_size=args.vector_size,
        window=args.window,
        min_count=args.min_count,
        workers=args.workers,
    )

    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    w2v.save(str(out_path))

    print(f"Word2Vec guardado en {out_path}")


if __name__ == "__main__":
    main()

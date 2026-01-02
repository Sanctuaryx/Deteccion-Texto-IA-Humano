from __future__ import annotations

import argparse
import json
import pathlib
from typing import List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import pandas as pd
from gensim.models import Word2Vec

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
    
from data.datasets import read_jsonl
from utils.logging import setup_logging
from models.custom_bilstm import BiLSTMClassifier
from models.dataset import TextDataset, TextExample, collate_batch
from models.vocab import Vocab
from utils.seed import set_seed


def load_examples(path: str) -> List[TextExample]:
    """
    Carga ejemplos desde:
      - CSV con columnas: text, generated
      - JSONL con campos: text, label
    Devuelve una lista de TextExample(text, label:int)
    """
    p = pathlib.Path(path)
    if p.suffix.lower() == ".csv":
        df = pd.read_csv(p)
        if "text" not in df.columns or "generated" not in df.columns:
            raise ValueError("CSV debe contener columnas 'text' y 'generated'.")
        examples: List[TextExample] = []
        for text, gen in zip(df["text"].tolist(), df["generated"].tolist()):
            # generated: 0.0 / 1.0 → int
            label = int(round(float(gen)))
            examples.append(TextExample(text=str(text), label=label))
        return examples
    else:
        rows = read_jsonl(p)
        return [TextExample(text=r["text"], label=int(r["label"])) for r in rows]


def train_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for input_ids, lengths, labels in tqdm(loader, desc="Train", leave=False):
        input_ids = input_ids.to(device)
        lengths = lengths.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(input_ids, lengths)  # (B,)
        loss = criterion(logits, labels)
        loss.backward()

        # Gradient clipping por seguridad
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(
    model,
    loader,
    criterion,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for input_ids, lengths, labels in tqdm(loader, desc="Val", leave=False):
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(input_ids, lengths)
            loss = criterion(logits, labels)

            total_loss += loss.item() * input_ids.size(0)
            probs = torch.sigmoid(logits)
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "loss": float(total_loss / len(loader.dataset)),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
        "roc_auc": float(roc_auc_score(labels, probs)),
    }
    return metrics, probs, labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/train.csv",
        help="Ruta al dataset de entrenamiento (CSV o JSONL)",
    )
    parser.add_argument(
        "--val_path",
        type=str,
        default="data/val.csv",
        help="Ruta al dataset de validación (CSV o JSONL)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=1)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--w2v_path", type=str, default=None, help="Ruta a un modelo Word2Vec de gensim para inicializar embeddings (opcional).")
    parser.add_argument("--out_dir", type=str, default="models/bilstm_rand/custom_model")
    parser.add_argument("--patience", type=int, default=3)
    
    args = parser.parse_args()

    setup_logging()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)

    # 1) Cargar datos
    train_examples = load_examples(args.train_path)
    val_examples = load_examples(args.val_path)

    # 2) Construir vocabulario (solo con train)
    train_texts = [ex.text for ex in train_examples]
    vocab = Vocab.build(train_texts, min_freq=args.min_freq)
    
    pretrained_tensor = None
    embed_dim = args.embed_dim  # valor por defecto

    if args.w2v_path is not None:
        print(f"Cargando Word2Vec desde {args.w2v_path}...")
        w2v = Word2Vec.load(args.w2v_path)
        embed_dim = w2v.vector_size

        # Matriz de embeddings: (vocab_size, embed_dim)
        emb_matrix = np.random.normal(
            scale=0.1, size=(len(vocab.itos), embed_dim)
        ).astype(np.float32)

        for idx, token in enumerate(vocab.itos):
            if token in w2v.wv:
                emb_matrix[idx] = w2v.wv[token]

        pretrained_tensor = torch.from_numpy(emb_matrix)
    else:
        print("Sin Word2Vec: usando embeddings aleatorios.")
    
    # 3) Crear datasets
    train_ds = TextDataset(train_examples, vocab=vocab, max_len=args.max_len)
    val_ds = TextDataset(val_examples, vocab=vocab, max_len=args.max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_batch(b, pad_index=vocab.pad_index),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda b: collate_batch(b, pad_index=vocab.pad_index),
    )

    # 4) Crear modelo
    model = BiLSTMClassifier(
        vocab_size=len(vocab.itos),
        embed_dim=embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        pad_index=vocab.pad_index,
        dropout=0.3,
        pretrained_embeddings=pretrained_tensor,
        freeze_embeddings=False,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    best_roc = 0.0
    best_probs = None
    best_labels = None
    patience = args.patience
    epochs_without_improvement = 0

    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        metrics, probs, labels = eval_epoch(model, val_loader, criterion, device)
        print(
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {metrics['loss']:.4f} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f}"
        )

        if metrics["roc_auc"] > best_roc:
            best_roc = metrics["roc_auc"]
            best_probs = probs
            best_labels = labels
            epochs_without_improvement = 0
            print("Mejor ROC-AUC, guardando modelo y vocabulario...")

            torch.save(model.state_dict(), out_dir / "model.pt")
            with (out_dir / "vocab.json").open("w", encoding="utf-8") as f:
                json.dump(
                    {
                        "itos": vocab.itos,
                        "pad_index": vocab.pad_index,
                        "unk_index": vocab.unk_index,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping por falta de mejora en ROC-AUC.")
                break

    # 5) Búsqueda de mejor umbral según F1 (si tenemos probs/labels)
    if best_probs is not None and best_labels is not None:
        thresholds = np.linspace(0.1, 0.9, 17)
        best_threshold = 0.5
        best_f1 = 0.0
        for thr in thresholds:
            preds = (best_probs >= thr).astype(int)
            f1 = f1_score(best_labels, preds)
            if f1 > best_f1 or (f1 == best_f1 and abs(thr - 0.5) < abs(best_threshold - 0.5)):
                best_f1 = f1
                best_threshold = float(thr)
        print(
            f"Mejor umbral en validación: {best_threshold:.2f} "
            f"(F1={best_f1:.3f}, ROC-AUC={best_roc:.3f})"
        )
    else:
        best_threshold = 0.5
        print("No se han guardado probs/labels; se usa threshold por defecto 0.5.")

    # 6) Guardar config.json con hiperparámetros y umbral
    config = {
        "embed_dim": args.embed_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "max_len": args.max_len,
        "min_freq": args.min_freq,
        "threshold": best_threshold,
        "vocab_size": len(vocab.itos),
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print("Entrenamiento completado. Modelo y configuración guardados en:", out_dir)


if __name__ == "__main__":
    main()

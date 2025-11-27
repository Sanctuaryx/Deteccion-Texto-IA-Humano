from __future__ import annotations

import argparse
import json
import pathlib
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

import numpy as np
from sklearn.metrics import f1_score
from iatd.data.datasets import read_jsonl
from iatd.logging import setup_logging
from iatd.models.custom_bilstm import BiLSTMClassifier
from iatd.models.dataset import TextDataset, TextExample, collate_batch
from iatd.models.vocab import Vocab
from iatd.utils.seed import set_seed


def load_examples(path: str) -> List[TextExample]:
    rows = read_jsonl(path)
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
        optimizer.step()

        total_loss += loss.item() * input_ids.size(0)
    return total_loss / len(loader.dataset)


def eval_epoch(model, loader, criterion, device: torch.device) -> dict:
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

    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "loss": total_loss / len(loader.dataset),
        "accuracy": float(accuracy_score(labels, preds)),
        "f1": float(f1_score(labels, preds)),
        "roc_auc": float(roc_auc_score(labels, probs)),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_path", type=str, default="data/train.jsonl"
    )
    parser.add_argument(
        "--val_path", type=str, default="data/val.jsonl"
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out_dir", type=str, default="artifacts/custom_model"
    )
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
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        pad_index=vocab.pad_index,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_roc = 0.0
    patience = 3
    epochs_without_improvement = 0

    for epoch in range(1, args.epochs + 1):
        print(f"Epoch {epoch}/{args.epochs}")
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        metrics = eval_epoch(model, val_loader, criterion, device)
        print(
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {metrics['loss']:.4f} | "
            f"Acc: {metrics['accuracy']:.4f} | "
            f"F1: {metrics['f1']:.4f} | "
            f"ROC-AUC: {metrics['roc_auc']:.4f}"
        )

        if metrics["roc_auc"] > best_roc:
            best_roc = metrics["roc_auc"]
            epochs_without_improvement = 0
            print("Mejor ROC-AUC, guardando modelo...")
            out_dir = pathlib.Path(args.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), out_dir / "model.pt")
            # guardar vocabulario
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


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()  # etiquetas 0/1

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


class MLPClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # salida binaria (logit)
        )

    def forward(self, x):
        logits = self.net(x).squeeze(1)  # (B,)
        return logits


def load_npz(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    X = data["X"]
    y = data["y"]
    return X, y


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X, y in tqdm(loader, desc="Train", leave=False):
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    return total_loss / len(loader.dataset)


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X, y in tqdm(loader, desc="Val/Test", leave=False):
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            loss = criterion(logits, y)

            total_loss += loss.item() * X.size(0)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    preds = (probs >= 0.5).astype(int)  # umbral 0.5 provisional

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
        "--train_npz",
        type=str,
        default="data/bert/bert_train.npz",
        help="NPZ con X,y de train (embeddings BERT).",
    )
    parser.add_argument(
        "--val_npz",
        type=str,
        default="data/bert/bert_val.npz",
        help="NPZ con X,y de validación.",
    )
    parser.add_argument(
        "--test_npz",
        type=str,
        default="data/bert/bert_test.npz",
        help="NPZ con X,y de test (opcional pero recomendado).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="artifacts/bert_mlp",
        help="Directorio donde guardar modelo y config.",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()

    train_npz = Path(args.train_npz)
    val_npz = Path(args.val_npz)
    test_npz = Path(args.test_npz)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ===== seleccionar dispositivo =====
    if torch.cuda.is_available():
        device = torch.device("cuda")
        backend = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        backend = "mps"
    else:
        device = torch.device("cpu")
        backend = "cpu"
    print(f"Usando dispositivo: {device} (backend: {backend})")

    # ===== cargar datos =====
    X_train, y_train = load_npz(train_npz)
    X_val, y_val = load_npz(val_npz)
    print(f"Train: {X_train.shape}, Val: {X_val.shape}")

    input_dim = X_train.shape[1]

    train_ds = NumpyDataset(X_train, y_train)
    val_ds = NumpyDataset(X_val, y_val)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False
    )

    # ===== modelo, loss, optim =====
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        dropout=0.3,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)

    best_roc = 0.0
    best_probs = None
    best_labels = None
    patience = args.patience
    epochs_without_improvement = 0

    # ===== entrenamiento con early stopping =====
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
            print("Mejor ROC-AUC en validación, guardando modelo...")
            torch.save(model.state_dict(), out_dir / "model.pt")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping por falta de mejora en ROC-AUC.")
                break

    # ===== búsqueda de umbral óptimo en validación =====
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
        print("No hay probs/labels suficientes; se usa threshold 0.5.")

    # ===== evaluación en test (si existe) =====
    if test_npz.exists():
        print("\nEvaluando en test con el mejor modelo y umbral óptimo...")
        # recargar mejor modelo
        best_model = MLPClassifier(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            dropout=0.3,
        ).to(device)
        state_dict = torch.load(out_dir / "model.pt", map_location=device)
        best_model.load_state_dict(state_dict)

        X_test, y_test = load_npz(test_npz)
        test_ds = NumpyDataset(X_test, y_test)
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False
        )

        _, probs_test, labels_test = eval_epoch(
            best_model, test_loader, criterion, device
        )
        preds_test = (probs_test >= best_threshold).astype(int)

        acc = accuracy_score(labels_test, preds_test)
        f1 = f1_score(labels_test, preds_test)
        roc = roc_auc_score(labels_test, probs_test)
        prec, rec, _, _ = precision_recall_fscore_support(
            labels_test, preds_test, average="binary", zero_division=0
        )
        cm = confusion_matrix(labels_test, preds_test)
        report = classification_report(labels_test, preds_test, digits=3)

        print("\n===== MÉTRICAS EN TEST (BERT+MLP) =====")
        print(f"Nº ejemplos: {len(labels_test)}")
        print(f"Umbral usado: {best_threshold:.2f}")
        print(f"Accuracy   : {acc:.4f}")
        print(f"F1         : {f1:.4f}")
        print(f"Precision  : {prec:.4f}")
        print(f"Recall     : {rec:.4f}")
        print(f"ROC-AUC    : {roc:.4f}")
        print("\nMatriz de confusión [ [TN, FP], [FN, TP] ]:")
        print(cm)
        print("\nClassification report:\n")
        print(report)

        metrics_test = {
            "num_examples": int(len(labels_test)),
            "threshold_used": float(best_threshold),
            "accuracy": float(acc),
            "f1": float(f1),
            "precision": float(prec),
            "recall": float(rec),
            "roc_auc": float(roc),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }
    else:
        print("No se ha encontrado test_npz; no se evalúa en test.")
        metrics_test = None

    # ===== guardar config =====
    config = {
        "input_dim": int(input_dim),
        "hidden_dim": int(args.hidden_dim),
        "threshold": float(best_threshold),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "patience": args.patience,
    }
    with (out_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    if metrics_test is not None:
        with (out_dir / "test_metrics.json").open("w", encoding="utf-8") as f:
            json.dump(metrics_test, f, ensure_ascii=False, indent=2)

    print("\nEntrenamiento BERT+MLP completado. Artefactos en:", out_dir)


if __name__ == "__main__":
    main()

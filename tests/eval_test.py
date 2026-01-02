from __future__ import annotations

import argparse
import json
import pathlib
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

# ==============================================
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import logic.models.custom_bilstm as bilstm
import logic.models.dataset as dataset
import logic.models.vocab as v
# ==============================================


# ========= PARTE 1: utilidades BiLSTM  =========

def load_test_examples(path: str) -> List[dataset.TextExample]:
    """
    Carga ejemplos de test desde un CSV con columnas:
      - text
      - generated (0 = humano, 1 = IA)
    """
    p = pathlib.Path(path)
    if p.suffix.lower() != ".csv":
        raise ValueError(
            f"Solo se soporta CSV en eval_test por ahora. Recibido: {p.suffix}"
        )

    df = pd.read_csv(p)
    if "text" not in df.columns or "generated" not in df.columns:
        raise ValueError("El CSV debe tener columnas 'text' y 'generated'.")

    examples: List[dataset.TextExample] = []
    for text, gen in zip(df["text"].tolist(), df["generated"].tolist()):
        label = int(round(float(gen)))
        examples.append(dataset.TextExample(text=str(text), label=label))

    return examples


def load_vocab_and_config(model_dir: pathlib.Path) -> Tuple[v.Vocab, dict]:
    with (model_dir / "vocab.json").open("r", encoding="utf-8") as f:
        vocab_cfg = json.load(f)

    vocab = v.Vocab(
        stoi={tok: i for i, tok in enumerate(vocab_cfg["itos"])},
        itos=vocab_cfg["itos"],
        pad_index=vocab_cfg["pad_index"],
        unk_index=vocab_cfg["unk_index"],
    )

    with (model_dir / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    return vocab, cfg


def evaluate_bilstm(
    model_dir: pathlib.Path,
    test_path: str,
    batch_size: int,
    device: torch.device,
) -> Dict:
    """
    Evalúa un modelo BiLSTM (con vocab.json, config.json, model.pt)
    en el CSV de test (text,generated).
    """
    print(f"\n=== Evaluando BiLSTM en {model_dir} ===")

    vocab, cfg = load_vocab_and_config(model_dir)
    threshold = float(cfg.get("threshold", 0.5))
    max_len = int(cfg.get("max_len", 256))
    embed_dim = int(cfg.get("embed_dim", 256))
    hidden_dim = int(cfg.get("hidden_dim", 256))
    num_layers = int(cfg.get("num_layers", 1))

    print(
        f"Config cargada: threshold={threshold}, max_len={max_len}, "
        f"embed_dim={embed_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}"
    )

    # Modelo
    model = bilstm.BiLSTMClassifier(
        vocab_size=len(vocab.itos),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        pad_index=vocab.pad_index,
        dropout=0.3,
    ).to(device)

    state_dict = torch.load(model_dir / "model.pt", map_location=device)
    model.load_state_dict(state_dict)

    # Dataset de test
    test_examples = load_test_examples(test_path)
    print(f"Ejemplos de test cargados: {len(test_examples)}")

    test_ds = dataset.TextDataset(test_examples, vocab=vocab, max_len=max_len)
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: dataset.collate_batch(b, pad_index=vocab.pad_index),
    )

    # Evaluación
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for input_ids, lengths, labels in tqdm(test_loader, desc="Test BiLSTM", leave=False):
            input_ids = input_ids.to(device)
            lengths = lengths.to(device)
            labels = labels.to(device)

            logits = model(input_ids, lengths)  # (B,)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    roc = roc_auc_score(labels, probs)
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, digits=3)

    metrics = {
        "num_examples": int(len(labels)),
        "threshold_used": float(threshold),
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": float(roc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    print("\n===== MÉTRICAS EN TEST (BiLSTM) =====")
    print(f"Nº ejemplos: {metrics['num_examples']}")
    print(f"Umbral usado: {metrics['threshold_used']:.2f}")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"F1         : {metrics['f1']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print(f"ROC-AUC    : {metrics['roc_auc']:.4f}")
    print("\nMatriz de confusión [ [TN, FP], [FN, TP] ]:")
    print(np.array(metrics["confusion_matrix"]))
    print("\nClassification report:\n")
    print(metrics["classification_report"])

    # Guardar métricas en JSON dentro del propio model_dir
    out_path = model_dir / "test_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nMétricas guardadas en {out_path}")

    return metrics


# ========= PARTE 2: utilidades BERT+MLP =========

class NumpyDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()

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
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        logits = self.net(x).squeeze(1)
        return logits


def load_npz(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    return data["X"], data["y"]


def evaluate_bert_mlp(
    model_dir: pathlib.Path,
    test_npz: pathlib.Path,
    batch_size: int,
    device: torch.device,
) -> Dict:
    """
    Evalúa el modelo BERT+MLP usando:
      - model_dir: contiene model.pt y config.json
      - test_npz: .npz con X (embeddings) e y (labels)
    Usa el threshold guardado en config.json.
    """
    print(f"\n=== Evaluando BERT+MLP en {model_dir} ===")

    # Cargar config (input_dim, hidden_dim, threshold, etc.)
    with (model_dir / "config.json").open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    input_dim = int(cfg["input_dim"])
    hidden_dim = int(cfg.get("hidden_dim", 256))
    threshold = float(cfg.get("threshold", 0.5))

    print(
        f"Config BERT+MLP: input_dim={input_dim}, hidden_dim={hidden_dim}, "
        f"threshold={threshold}"
    )

    # Cargar modelo
    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=0.3,
    ).to(device)

    state_dict = torch.load(model_dir / "model.pt", map_location=device)
    model.load_state_dict(state_dict)

    # Cargar datos de test
    X_test, y_test = load_npz(test_npz)
    print(f"Test NPZ cargado: X={X_test.shape}, y={y_test.shape}")
    test_ds = NumpyDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Evaluar
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Test BERT+MLP", leave=False):
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            probs = torch.sigmoid(logits)

            all_probs.append(probs.cpu())
            all_labels.append(y.cpu())

    probs = torch.cat(all_probs).numpy()
    labels = torch.cat(all_labels).numpy()

    roc = roc_auc_score(labels, probs)
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    precision, recall, _, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )

    cm = confusion_matrix(labels, preds)
    report = classification_report(labels, preds, digits=3)

    metrics = {
        "num_examples": int(len(labels)),
        "threshold_used": float(threshold),
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "roc_auc": float(roc),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    print("\n===== MÉTRICAS EN TEST (BERT+MLP) =====")
    print(f"Nº ejemplos: {metrics['num_examples']}")
    print(f"Umbral usado: {metrics['threshold_used']:.2f}")
    print(f"Accuracy   : {metrics['accuracy']:.4f}")
    print(f"F1         : {metrics['f1']:.4f}")
    print(f"Precision  : {metrics['precision']:.4f}")
    print(f"Recall     : {metrics['recall']:.4f}")
    print(f"ROC-AUC    : {metrics['roc_auc']:.4f}")
    print("\nMatriz de confusión [ [TN, FP], [FN, TP] ]:")
    print(np.array(metrics["confusion_matrix"]))
    print("\nClassification report:\n")
    print(metrics["classification_report"])

    out_path = model_dir / "test_metrics.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\nMétricas guardadas en {out_path}")

    return metrics


# ========= PARTE 3: orquestador y comparación =========

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test_path",
        type=str,
        default="data/test_es.csv",
        help="CSV de test (text,generated) para los modelos BiLSTM.",
    )
    parser.add_argument(
        "--bilstm_rand_dir",
        type=str,
        default=None,
        help="Directorio del modelo BiLSTM base (embeddings aleatorios).",
    )
    parser.add_argument(
        "--bilstm_w2v_dir",
        type=str,
        default=None,
        help="Directorio del modelo BiLSTM + Word2Vec.",
    )
    parser.add_argument(
        "--bilstm_other_dir",
        type=str,
        default=None,
        help="Opcional: otro BiLSTM (por ejemplo el primero que entrenaste).",
    )
    parser.add_argument(
        "--bert_mlp_dir",
        type=str,
        default=None,
        help="Directorio del modelo BERT+MLP (model.pt, config.json).",
    )
    parser.add_argument(
        "--bert_test_npz",
        type=str,
        default=None,
        help="Ruta al .npz de test para BERT+MLP (X,y).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size para evaluación.",
    )
    args = parser.parse_args()

    # dispositivo
    if torch.cuda.is_available():
        device = torch.device("cuda")
        backend = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        backend = "mps"
    else:
        device = torch.device("cpu")
        backend = "cpu"
    print(f"Usando dispositivo para test: {device} (backend: {backend})")

    all_results: Dict[str, Dict] = {}

    # Evaluar BiLSTM base
    if args.bilstm_rand_dir is not None:
        mdir = pathlib.Path(args.bilstm_rand_dir)
        all_results["bilstm_rand"] = evaluate_bilstm(
            mdir, args.test_path, args.batch_size, device
        )

    # Evaluar BiLSTM + Word2Vec
    if args.bilstm_w2v_dir is not None:
        mdir = pathlib.Path(args.bilstm_w2v_dir)
        all_results["bilstm_w2v"] = evaluate_bilstm(
            mdir, args.test_path, args.batch_size, device
        )

    # Evaluar tercer BiLSTM opcional
    if args.bilstm_other_dir is not None:
        mdir = pathlib.Path(args.bilstm_other_dir)
        all_results["bilstm_other"] = evaluate_bilstm(
            mdir, args.test_path, args.batch_size, device
        )

    # Evaluar BERT+MLP
    if args.bert_mlp_dir is not None and args.bert_test_npz is not None:
        mdir = pathlib.Path(args.bert_mlp_dir)
        tnpz = pathlib.Path(args.bert_test_npz)
        all_results["bert_mlp"] = evaluate_bert_mlp(
            mdir, tnpz, args.batch_size, device
        )

    # Resumen comparativo
    if all_results:
        print("\n\n===== RESUMEN COMPARATIVO (TEST) =====")
        print(
            f"{'Modelo':15} | {'Acc':6} | {'F1':6} | {'Prec':6} | {'Rec':6} | {'ROC-AUC':7} | {'Thr':5}"
        )
        print("-" * 72)
        for name, m in all_results.items():
            print(
                f"{name:15} | "
                f"{m['accuracy']:.4f} | "
                f"{m['f1']:.4f} | "
                f"{m['precision']:.4f} | "
                f"{m['recall']:.4f} | "
                f"{m['roc_auc']:.4f} | "
                f"{m['threshold_used']:.2f}"
            )

        # Guardar todo en un JSON global
        out_all = ROOT / "artifacts" / "all_models_metrics.json"
        out_all.parent.mkdir(parents=True, exist_ok=True)
        with out_all.open("w", encoding="utf-8") as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        print(f"\nMétricas de todos los modelos guardadas en {out_all}")
    else:
        print("No se ha evaluado ningún modelo (revisa los argumentos).")


if __name__ == "__main__":
    main()


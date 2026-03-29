
import argparse
import json
import math
import random
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from export_lstm_artifacts import export_artifacts

TEXT_COLS = [
    "Describe how this painting makes you feel.",
    "If this painting was a food, what would be?",
    "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.",
]

NUMERIC_COLS = [
    "On a scale of 1–10, how intense is the emotion conveyed by the artwork?",
    "How many prominent colours do you notice in this painting?",
    "How many objects caught your eye in the painting?",
    "How much (in Canadian dollars) would you be willing to pay for this painting?",
]

LIKERT_COLS = [
    "This art piece makes me feel sombre.",
    "This art piece makes me feel content.",
    "This art piece makes me feel calm.",
    "This art piece makes me feel uneasy.",
]

MULTI_COLS = [
    "If you could purchase this painting, which room would you put that painting in?",
    "If you could view this art in person, who would you want to view it with?",
    "What season does this art piece remind you of?",
]

TARGET_COL = "Painting"

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SEP_TOKEN = "[SEP]"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_text(s: str) -> str:
    s = "" if pd.isna(s) else str(s)
    s = s.lower()
    s = re.sub(rf"[{re.escape(string.punctuation)}]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def concat_text_fields(row: pd.Series) -> str:
    parts = [normalize_text(row.get(col, "")) for col in TEXT_COLS]
    return f" {SEP_TOKEN} ".join(parts)


def tokenize(text: str) -> list[str]:
    return text.split() if text else []


def build_vocab(texts: list[str], min_freq: int = 2, max_size: int = 5000) -> dict[str, int]:
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    items = [(tok, freq) for tok, freq in counter.items() if freq >= min_freq and tok != SEP_TOKEN]
    items.sort(key=lambda x: (-x[1], x[0]))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, SEP_TOKEN: 2}
    for tok, _ in items:
        if tok in vocab:
            continue
        if len(vocab) >= max_size:
            break
        vocab[tok] = len(vocab)
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int) -> np.ndarray:
    tokens = tokenize(text)
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens[:max_len]]
    if len(ids) < max_len:
        ids += [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return np.asarray(ids, dtype=np.int64)


def parse_money(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace("$", "").replace(",", "")
    try:
        return float(s)
    except ValueError:
        return np.nan


LIKERT_MAP = {
    "1 - strongly disagree": 1.0,
    "2 - disagree": 2.0,
    "3 - neutral/unsure": 3.0,
    "4 - agree": 4.0,
    "5 - strongly agree": 5.0,
}


def parse_likert(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower()
    return LIKERT_MAP.get(s, np.nan)


def fit_preprocessor(train_df: pd.DataFrame, max_vocab: int, min_freq: int, max_len: int):
    text_series = train_df.apply(concat_text_fields, axis=1)
    vocab = build_vocab(text_series.tolist(), min_freq=min_freq, max_size=max_vocab)

    numeric_stats = {}
    for col in NUMERIC_COLS:
        vals = train_df[col].map(parse_money if "Canadian dollars" in col else lambda x: pd.to_numeric(x, errors="coerce"))
        mean = float(vals.mean()) if not np.isnan(vals.mean()) else 0.0
        std = float(vals.std(ddof=0)) if not np.isnan(vals.std(ddof=0)) and vals.std(ddof=0) > 0 else 1.0
        numeric_stats[col] = {"mean": mean, "std": std}

    likert_stats = {}
    for col in LIKERT_COLS:
        vals = train_df[col].map(parse_likert)
        mode_series = vals.dropna().mode()
        fill = float(mode_series.iloc[0]) if not mode_series.empty else 3.0
        filled = vals.fillna(fill)
        mean = float(filled.mean()) if not np.isnan(filled.mean()) else fill
        std = float(filled.std(ddof=0)) if not np.isnan(filled.std(ddof=0)) and filled.std(ddof=0) > 0 else 1.0
        likert_stats[col] = {"fill": fill, "mean": mean, "std": std}

    multi_categories = {}
    for col in MULTI_COLS:
        vals = set()
        for x in train_df[col].dropna().astype(str):
            for part in x.split(","):
                part = part.strip()
                if part:
                    vals.add(part)
        multi_categories[col] = sorted(vals)

    label_names = sorted(train_df[TARGET_COL].dropna().astype(str).unique().tolist())

    meta = {
        "vocab": vocab,
        "max_len": max_len,
        "numeric_stats": numeric_stats,
        "likert_stats": likert_stats,
        "multi_categories": multi_categories,
        "label_names": label_names,
    }
    return meta


def transform_structured(df: pd.DataFrame, meta: dict) -> np.ndarray:
    feats = []

    for col in NUMERIC_COLS:
        vals = df[col].map(parse_money if "Canadian dollars" in col else lambda x: pd.to_numeric(x, errors="coerce")).astype(float)
        mean = meta["numeric_stats"][col]["mean"]
        std = meta["numeric_stats"][col]["std"]
        vals = vals.fillna(mean)
        feats.append(((vals - mean) / std).to_numpy(dtype=np.float32)[:, None])

    for col in LIKERT_COLS:
        vals = df[col].map(parse_likert).astype(float)
        fill = meta["likert_stats"][col]["fill"]
        mean = meta["likert_stats"][col]["mean"]
        std = meta["likert_stats"][col]["std"]
        vals = vals.fillna(fill)
        feats.append(((vals - mean) / std).to_numpy(dtype=np.float32)[:, None])

    for col in MULTI_COLS:
        cats = meta["multi_categories"][col]
        block = np.zeros((len(df), len(cats)), dtype=np.float32)
        cat_to_idx = {c: i for i, c in enumerate(cats)}
        for i, x in enumerate(df[col].fillna("").astype(str)):
            for part in x.split(","):
                part = part.strip()
                if part in cat_to_idx:
                    block[i, cat_to_idx[part]] = 1.0
        feats.append(block)

    return np.concatenate(feats, axis=1)


def transform_text(df: pd.DataFrame, meta: dict) -> np.ndarray:
    texts = df.apply(concat_text_fields, axis=1)
    seqs = [encode_text(text, meta["vocab"], meta["max_len"]) for text in texts]
    return np.stack(seqs)


def encode_labels(labels: pd.Series, label_names: list[str]) -> np.ndarray:
    lab_to_idx = {lab: i for i, lab in enumerate(label_names)}
    return labels.astype(str).map(lab_to_idx).to_numpy(dtype=np.int64)


def group_split(df: pd.DataFrame, group_col: str = "unique_id", train_frac: float = 0.7, val_frac: float = 0.15, seed: int = 0):
    groups = df[group_col].dropna().unique().tolist()
    rng = np.random.default_rng(seed)
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(round(train_frac * n))
    n_val = int(round(val_frac * n))

    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train:n_train + n_val])
    test_groups = set(groups[n_train + n_val:])

    train_df = df[df[group_col].isin(train_groups)].copy()
    val_df = df[df[group_col].isin(val_groups)].copy()
    test_df = df[df[group_col].isin(test_groups)].copy()
    return train_df, val_df, test_df


class MultimodalDataset(Dataset):
    def __init__(self, seqs, structured, labels):
        self.seqs = torch.tensor(seqs, dtype=torch.long)
        self.structured = torch.tensor(structured, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.structured[idx], self.labels[idx]


class MultimodalLSTM(nn.Module):
    def __init__(self, vocab_size, structured_dim, emb_dim=100, hidden_dim=128, dropout=0.3, num_classes=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=emb_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.structured_fc = nn.Linear(structured_dim, 64)
        self.fusion_fc = nn.Linear(hidden_dim + 64, 128)
        self.output_fc = nn.Linear(128, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, seqs, structured):
        emb = self.embedding(seqs)
        _, (h_n, _) = self.lstm(emb)
        text_repr = self.dropout(h_n[-1])

        structured_repr = self.relu(self.structured_fc(structured))
        fused = torch.cat([text_repr, structured_repr], dim=1)
        fused = self.dropout(self.relu(self.fusion_fc(fused)))
        return self.output_fc(fused)


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for seqs, structured, labels in loader:
            seqs = seqs.to(device)
            structured = structured.to(device)
            labels = labels.to(device)

            logits = model(seqs, structured)
            loss = criterion(logits, labels)

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size
            correct += (logits.argmax(dim=1) == labels).sum().item()

    return total_loss / max(total, 1), correct / max(total, 1)


def train_one_config(train_df, val_df, meta, args, config, device):
    X_train_text = transform_text(train_df, meta)
    X_val_text = transform_text(val_df, meta)
    X_train_struct = transform_structured(train_df, meta)
    X_val_struct = transform_structured(val_df, meta)
    y_train = encode_labels(train_df[TARGET_COL], meta["label_names"])
    y_val = encode_labels(val_df[TARGET_COL], meta["label_names"])

    train_ds = MultimodalDataset(X_train_text, X_train_struct, y_train)
    val_ds = MultimodalDataset(X_val_text, X_val_struct, y_val)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    model = MultimodalLSTM(
        vocab_size=len(meta["vocab"]),
        structured_dim=X_train_struct.shape[1],
        emb_dim=config["emb_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
        num_classes=len(meta["label_names"]),
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=args.weight_decay,
    )

    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1
    patience_left = args.patience
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total = 0

        for seqs, structured, labels in train_loader:
            seqs = seqs.to(device)
            structured = structured.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(seqs, structured)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total += batch_size

        train_loss = total_loss / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_acc": val_acc})

        print(
            f"[cfg={config}] epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}.")
                break

    model.load_state_dict(best_state)
    return {
        "model": model,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "history": history,
        "config": config,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default="lstm_run")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_vocab", type=int, default=5000)
    parser.add_argument("--min_freq", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=64)
    parser.add_argument("--hidden_sizes", type=int, nargs="+", default=[64, 128])
    parser.add_argument("--emb_dims", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--dropouts", type=float, nargs="+", default=[0.2, 0.3])
    parser.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 5e-4])
    args = parser.parse_args()

    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)

    train_df, val_df, test_df = group_split(df, group_col="unique_id", train_frac=0.7, val_frac=0.15, seed=args.seed)
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}")

    meta = fit_preprocessor(train_df, max_vocab=args.max_vocab, min_freq=args.min_freq, max_len=args.max_len)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    all_configs = []
    for hidden_dim in args.hidden_sizes:
        for emb_dim in args.emb_dims:
            for dropout in args.dropouts:
                for lr in args.lrs:
                    all_configs.append({
                        "hidden_dim": hidden_dim,
                        "emb_dim": emb_dim,
                        "dropout": dropout,
                        "lr": lr,
                    })

    best_run = None
    for cfg in all_configs:
        run = train_one_config(train_df, val_df, meta, args, cfg, device)
        if best_run is None or run["best_val_loss"] < best_run["best_val_loss"]:
            best_run = run

    print("\nBEST CONFIG:", best_run["config"])
    print("BEST VAL LOSS:", best_run["best_val_loss"])
    print("BEST EPOCH:", best_run["best_epoch"])

    # Evaluate on held-out test split
    X_test_text = transform_text(test_df, meta)
    X_test_struct = transform_structured(test_df, meta)
    y_test = encode_labels(test_df[TARGET_COL], meta["label_names"])
    test_ds = MultimodalDataset(X_test_text, X_test_struct, y_test)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(best_run["model"], test_loader, criterion, device)
    print(f"TEST LOSS={test_loss:.4f} TEST ACC={test_acc:.4f}")

    # Save torch checkpoint
    torch.save(
        {
            "state_dict": best_run["model"].state_dict(),
            "meta": meta,
            "config": best_run["config"],
            "test_loss": test_loss,
            "test_acc": test_acc,
        },
        out_dir / "best_lstm_model.pt",
    )

    # Save summary
    with open(out_dir / "training_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_config": best_run["config"],
                "best_val_loss": best_run["best_val_loss"],
                "best_epoch": best_run["best_epoch"],
                "test_loss": test_loss,
                "test_acc": test_acc,
            },
            f,
            indent=2,
        )

    # Export NumPy artifacts for pred.py
    export_artifacts(best_run["model"], meta, out_dir=out_dir)
    print(f"Saved everything to {out_dir}")


if __name__ == "__main__":
    main()

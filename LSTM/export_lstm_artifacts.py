import json
import re
import string
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MAX_LEN = 64
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
SEP_TOKEN = "[SEP]"

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

_TRANSLATOR = str.maketrans({ch: " " for ch in string.punctuation})


def safe_text(x):
    return "" if pd.isna(x) else str(x)


def tokenize(text: str):
    return text.lower().translate(_TRANSLATOR).split()


def parse_likert(x):
    if pd.isna(x):
        return np.nan
    m = re.search(r"(\d+)", str(x))
    return float(m.group(1)) if m else np.nan


def parse_money_raw(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip().lower().replace(",", "")
    for token in ["canadian dollars", "dollars", "dollar", "cad", "$"]:
        s = s.replace(token, "")
    m = re.search(r"-?\d+(?:\.\d+)?", s)
    return float(m.group(0)) if m else np.nan


def fit_meta(train_df: pd.DataFrame, label_names):
    vocab_counter = Counter()
    for _, row in train_df.iterrows():
        joined = f" {SEP_TOKEN} ".join(safe_text(row[c]) for c in TEXT_COLS)
        vocab_counter.update(tokenize(joined))

    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1, SEP_TOKEN: 2}
    for tok, freq in vocab_counter.most_common():
        if freq < 2:
            continue
        if tok in vocab:
            continue
        if len(vocab) >= 5000:
            break
        vocab[tok] = len(vocab)

    numeric_stats = {}
    for col in NUMERIC_COLS[:3]:
        s = pd.to_numeric(train_df[col], errors="coerce")
        median = float(s.median())
        clip_min = float(s.quantile(0.01))
        clip_max = float(s.quantile(0.99))
        filled = s.fillna(median).clip(clip_min, clip_max)
        numeric_stats[col] = {
            "median": median,
            "clip_min": clip_min,
            "clip_max": clip_max,
            "mean": float(filled.mean()),
            "std": float(filled.std(ddof=0) if filled.std(ddof=0) > 0 else 1.0),
        }

    money = train_df[NUMERIC_COLS[3]].map(parse_money_raw)
    median_raw = float(money.median())
    clip_min_raw = float(money.quantile(0.01))
    clip_max_raw = float(money.quantile(0.99))
    money_filled = money.fillna(median_raw).clip(clip_min_raw, clip_max_raw)
    money_log = np.log1p(np.maximum(money_filled.to_numpy(dtype=float), 0.0))
    numeric_stats[NUMERIC_COLS[3]] = {
        "median_raw": median_raw,
        "clip_min_raw": clip_min_raw,
        "clip_max_raw": clip_max_raw,
        "mean": float(money_log.mean()),
        "std": float(money_log.std() if money_log.std() > 0 else 1.0),
    }

    likert_stats = {}
    for col in LIKERT_COLS:
        s = train_df[col].map(parse_likert)
        mode = float(s.mode(dropna=True).iloc[0])
        filled = s.fillna(mode)
        likert_stats[col] = {
            "mode": mode,
            "mean": float(filled.mean()),
            "std": float(filled.std(ddof=0) if filled.std(ddof=0) > 0 else 1.0),
        }

    multi_categories = {}
    for col in MULTI_COLS:
        vals = set()
        for x in train_df[col].dropna().astype(str):
            for part in x.split(","):
                part = part.strip()
                if part:
                    vals.add(part)
        multi_categories[col] = sorted(vals)

    return {
        "vocab": vocab,
        "numeric_stats": numeric_stats,
        "likert_stats": likert_stats,
        "multi_categories": multi_categories,
        "label_names": list(label_names),
    }


def export_artifacts(model, meta, out_dir="."):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "lstm_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f)

    state = model.state_dict()
    np_state = {k: v.detach().cpu().numpy() for k, v in state.items()}
    np.savez_compressed(out_dir / "lstm_params.npz", **np_state)
    print("Saved:", out_dir / "lstm_meta.json", "and", out_dir / "lstm_params.npz")
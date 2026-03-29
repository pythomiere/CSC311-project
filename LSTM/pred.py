import json
import re
import string
from pathlib import Path

import numpy as np
import pandas as pd

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
_MODEL_CACHE = None


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=1, keepdims=True)


def _tokenize(text: str):
    text = text.lower().translate(_TRANSLATOR)
    return text.split()


def _safe_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x)


def _parse_likert(x, default: float) -> float:
    if pd.isna(x):
        return default
    m = re.search(r"(\d+)", str(x))
    if m:
        return float(m.group(1))
    return default


def _parse_money(x, default: float, clip_min: float, clip_max: float) -> float:
    if pd.isna(x):
        value = default
    else:
        s = str(x).strip().lower()
        s = s.replace(",", "")
        s = s.replace("cad", "")
        s = s.replace("canadian dollars", "")
        s = s.replace("dollars", "")
        s = s.replace("dollar", "")
        s = s.replace("$", "")
        m = re.search(r"-?\d+(?:\.\d+)?", s)
        value = float(m.group(0)) if m else default
    value = min(max(value, clip_min), clip_max)
    return np.log1p(max(value, 0.0))


def _parse_numeric(x, default: float, clip_min: float, clip_max: float) -> float:
    if pd.isna(x):
        value = default
    else:
        try:
            value = float(x)
        except Exception:
            value = default
    return min(max(value, clip_min), clip_max)


def _encode_multiselect(value, categories):
    out = np.zeros(len(categories), dtype=np.float64)
    if pd.isna(value):
        return out
    present = {part.strip() for part in str(value).split(",") if part.strip()}
    for i, cat in enumerate(categories):
        if cat in present:
            out[i] = 1.0
    return out


def _build_text_sequence(row, vocab):
    parts = [_safe_text(row[c]) for c in TEXT_COLS]
    joined = f" {SEP_TOKEN} ".join(parts)
    tokens = _tokenize(joined)
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokens[:MAX_LEN]]
    length = len(ids)
    if length < MAX_LEN:
        ids.extend([vocab[PAD_TOKEN]] * (MAX_LEN - length))
    return np.array(ids, dtype=np.int64), length

def _prepare_structured(df: pd.DataFrame, meta: dict) -> np.ndarray:
    rows = []
    for _, row in df.iterrows():
        feats = []

        for col in NUMERIC_COLS[:3]:
            info = meta["numeric_stats"][col]
            raw = row[col]

            try:
                val = float(raw)
            except (TypeError, ValueError):
                val = info.get("mean", 0.0)

            std = info.get("std", 1.0)
            if std == 0:
                std = 1.0

            feats.append((val - info.get("mean", 0.0)) / std)

        money_info = meta["numeric_stats"][NUMERIC_COLS[3]]
        raw_money = row[NUMERIC_COLS[3]]

        if pd.isna(raw_money):
            money_val = money_info.get("mean", 0.0)
        else:
            s = str(raw_money)
            s = s.replace("$", "").replace(",", "").strip()
            try:
                money_val = float(s)
            except ValueError:
                money_val = money_info.get("mean", 0.0)

        std = money_info.get("std", 1.0)
        if std == 0:
            std = 1.0

        feats.append((money_val - money_info.get("mean", 0.0)) / std)

        for col in LIKERT_COLS:
            info = meta["likert_stats"][col]
            fallback = info.get("mode", 3.0)
            val = _parse_likert(row[col], fallback)

            std = info.get("std", 1.0)
            if std == 0:
                std = 1.0

            feats.append((val - info.get("mean", 0.0)) / std)

        for col in MULTI_COLS:
            feats.extend(_encode_multiselect(row[col], meta["multi_categories"][col]).tolist())

        rows.append(feats)

    return np.asarray(rows, dtype=np.float64)


def _lstm_forward(emb_seq, params):
    W_ih = params["lstm.weight_ih_l0"]
    W_hh = params["lstm.weight_hh_l0"]
    b_ih = params["lstm.bias_ih_l0"]
    b_hh = params["lstm.bias_hh_l0"]
    hidden_size = W_hh.shape[1]

    batch_size = emb_seq.shape[0]
    h = np.zeros((batch_size, hidden_size), dtype=np.float64)
    c = np.zeros((batch_size, hidden_size), dtype=np.float64)

    for t in range(emb_seq.shape[1]):
        x_t = emb_seq[:, t, :]
        gates = x_t @ W_ih.T + h @ W_hh.T + b_ih + b_hh
        i, f, g, o = np.split(gates, 4, axis=1)
        i = _sigmoid(i)
        f = _sigmoid(f)
        g = np.tanh(g)
        o = _sigmoid(o)
        c = f * c + i * g
        h = o * np.tanh(c)
    return h


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _load_model():
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    base = Path(__file__).resolve().parent
    with open(base / "lstm_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    params_npz = np.load(base / "lstm_params.npz")
    params = {k: params_npz[k].astype(np.float64) for k in params_npz.files}
    _MODEL_CACHE = (meta, params)
    return _MODEL_CACHE


def predict_all(filename: str):
    meta, params = _load_model()
    df = pd.read_csv(filename)

    seqs = []
    for _, row in df.iterrows():
        seq, _ = _build_text_sequence(row, meta["vocab"])
        seqs.append(seq)
    seqs = np.stack(seqs, axis=0)

    structured = _prepare_structured(df, meta)

    emb = params["embedding.weight"][seqs]
    text_repr = _lstm_forward(emb, params)

    struct_hidden = _relu(structured @ params["structured_fc.weight"].T + params["structured_fc.bias"])

    fused = np.concatenate([text_repr, struct_hidden], axis=1)
    fusion_hidden = _relu(fused @ params["fusion_fc.weight"].T + params["fusion_fc.bias"])
    logits = fusion_hidden @ params["output_fc.weight"].T + params["output_fc.bias"]

    probs = _softmax(logits)
    pred_ids = np.argmax(probs, axis=1)
    labels = [meta["label_names"][i] for i in pred_ids]
    return np.array(labels, dtype=object)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        raise SystemExit("Usage: python pred.py <csv_file>")
    preds = predict_all(sys.argv[1])
    for p in preds:
        print(p)


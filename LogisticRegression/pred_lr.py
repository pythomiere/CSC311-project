"""
Standalone logistic regression prediction script.
Imports only: standard library, numpy, pandas.

Requires: lr_model_params.json (preprocessing + LR weights)

Usage:
    from pred_lr import predict_all
    predictions = predict_all("test_data.csv")
"""

import json
import re
import os
import math
import numpy as np
import pandas as pd

# ── Load model parameters on import ──────────────────────────────────────────

_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(_DIR, 'lr_model_params.json'), 'r') as f:
    _PARAMS = json.load(f)

# ── Column names ─────────────────────────────────────────────────────────────

NUMERIC_COLS = [
    'On a scale of 1\u201310, how intense is the emotion conveyed by the artwork?',
    'How many prominent colours do you notice in this painting?',
    'How many objects caught your eye in the painting?',
]

LIKERT_COLS = [
    'This art piece makes me feel sombre.',
    'This art piece makes me feel content.',
    'This art piece makes me feel calm.',
    'This art piece makes me feel uneasy.',
]

DOLLAR_COL = 'How much (in Canadian dollars) would you be willing to pay for this painting?'

MULTI_SELECT_COLS = [
    'If you could purchase this painting, which room would you put that painting in?',
    'If you could view this art in person, who would you want to view it with?',
    'What season does this art piece remind you of?',
]

TEXT_COLS = [
    'Describe how this painting makes you feel.',
    'If this painting was a food, what would be?',
    'Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.',
]

SCALED_COLS = NUMERIC_COLS + LIKERT_COLS + ['dollar_log']


# ── Preprocessing helpers ────────────────────────────────────────────────────

def _extract_rating(response):
    if pd.isna(response):
        return np.nan
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else np.nan


def _parse_dollar(val):
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    s = re.sub(r'[\$,]', '', s)
    s = re.sub(r'(dollars|dollar|cad|cdn|ca|\bca\b)', '', s).strip()
    match = re.search(r'[\d]+\.?[\d]*', s)
    return float(match.group()) if match else np.nan


def _preprocess(df: pd.DataFrame) -> np.ndarray:
    """Full preprocessing pipeline. Returns dense feature matrix."""
    p = _PARAMS
    df = df.copy()

    # Likert parsing + imputation
    for col in LIKERT_COLS:
        df[col] = df[col].apply(_extract_rating)
        df[col] = df[col].fillna(p['likert_modes'][col])

    # Dollar parsing, clipping, log transform
    df['dollar_parsed'] = df[DOLLAR_COL].apply(_parse_dollar)
    df['dollar_parsed'] = df['dollar_parsed'].fillna(p['dollar_median'])
    df['dollar_parsed'] = df['dollar_parsed'].clip(p['dollar_p1'], p['dollar_p99'])
    df['dollar_log'] = np.log1p(df['dollar_parsed'])

    # Numeric clipping + imputation
    for col in NUMERIC_COLS:
        df[col] = df[col].fillna(p['numeric_medians'][col])
        df[col] = df[col].clip(
            p['numeric_clip'][col]['lo'],
            p['numeric_clip'][col]['hi'],
        )

    # Standardization
    scaler_mean = np.array(p['scaler_mean'])
    scaler_scale = np.array(p['scaler_scale'])
    X_scaled = (df[SCALED_COLS].values - scaler_mean) / scaler_scale

    # Multi-hot encoding
    multi_hot_cats = p['multi_hot_categories']
    mh_parts = []
    for col in MULTI_SELECT_COLS:
        cats = multi_hot_cats[col]
        mat = np.zeros((len(df), len(cats)), dtype=np.float64)
        for i, val in enumerate(df[col].values):
            if pd.isna(val):
                continue
            selected = {item.strip() for item in str(val).split(',')}
            for j, cat in enumerate(cats):
                if cat in selected:
                    mat[i, j] = 1.0
        mh_parts.append(mat)
    X_mh = np.hstack(mh_parts)

    # TF-IDF (manual implementation matching sklearn's TfidfVectorizer)
    tfidf_parts = []
    for col in TEXT_COLS:
        vocab = p['tfidf'][col]['vocabulary']
        idf = np.array(p['tfidf'][col]['idf'])
        n_features = len(idf)

        mat = np.zeros((len(df), n_features), dtype=np.float64)
        for i, text in enumerate(df[col].fillna('').astype(str).values):
            tokens = re.findall(r'(?u)\b\w\w+\b', text.lower())

            tf = {}
            for token in tokens:
                if token in vocab:
                    idx = vocab[token]
                    tf[idx] = tf.get(idx, 0) + 1

            for idx, count in tf.items():
                mat[i, idx] = (1.0 + math.log(count)) * idf[idx]

            norm = np.sqrt(np.sum(mat[i] ** 2))
            if norm > 0:
                mat[i] /= norm

        tfidf_parts.append(mat)

    X_tfidf = np.hstack(tfidf_parts)

    return np.hstack([X_scaled, X_mh, X_tfidf])


# ── Logistic regression prediction ──────────────────────────────────────────

def _lr_predict(X: np.ndarray) -> list[str]:
    """Multinomial logistic regression: argmax(X @ W^T + b)."""
    lr = _PARAMS['lr']
    coef = np.array(lr['coef'])         # (n_classes, n_features)
    intercept = np.array(lr['intercept'])  # (n_classes,)
    classes = np.array(lr['classes'])

    logits = X @ coef.T + intercept
    pred_idx = np.argmax(logits, axis=1)
    return classes[pred_idx].tolist()


# ── Public API ───────────────────────────────────────────────────────────────

def predict_all(filename: str) -> list[str]:
    """
    Make predictions for the data in filename.

    Parameters
    ----------
    filename : str
        Path to a CSV file with the same columns as the training data.

    Returns
    -------
    list[str]
        List of painting name predictions.
    """
    df = pd.read_csv(filename)
    X = _preprocess(df)
    return _lr_predict(X)


if __name__ == '__main__':
    import sys
    fname = sys.argv[1] if len(sys.argv) > 1 else 'data/training_data_202601.csv'
    preds = predict_all(fname)
    print(f"Made {len(preds)} predictions")
    for name in sorted(set(preds)):
        count = preds.count(name)
        print(f"  {name}: {count}")

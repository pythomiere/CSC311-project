"""
Standalone SVM prediction script.
Imports only: standard library, numpy, pandas.

Requires:
  - svm_model_params.json  (preprocessing + SVM config)
  - svm_model_arrays.npz   (support vectors + dual coefficients)

Usage:
    from pred import predict_all
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

with open(os.path.join(_DIR, 'svm_model_params.json'), 'r') as f:
    _PARAMS = json.load(f)

_ARRAYS = np.load(os.path.join(_DIR, 'svm_model_arrays.npz'))
_SUPPORT_VECTORS = _ARRAYS['support_vectors']   # (n_sv, n_features)
_DUAL_COEF = _ARRAYS['dual_coef']               # (n_classes-1, n_sv)

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
        vocab = p['tfidf'][col]['vocabulary']  # word -> index
        idf = np.array(p['tfidf'][col]['idf'])  # (n_features,)
        n_features = len(idf)

        mat = np.zeros((len(df), n_features), dtype=np.float64)
        for i, text in enumerate(df[col].fillna('').astype(str).values):
            # Tokenize: lowercase, extract word tokens (matching sklearn default)
            tokens = re.findall(r'(?u)\b\w\w+\b', text.lower())

            # Term frequency
            tf = {}
            for token in tokens:
                if token in vocab:
                    idx = vocab[token]
                    tf[idx] = tf.get(idx, 0) + 1

            # Sublinear TF: 1 + log(tf)
            for idx, count in tf.items():
                mat[i, idx] = (1.0 + math.log(count)) * idf[idx]

            # L2 normalize the row
            norm = np.sqrt(np.sum(mat[i] ** 2))
            if norm > 0:
                mat[i] /= norm

        tfidf_parts.append(mat)

    X_tfidf = np.hstack(tfidf_parts)

    # Combine all features
    return np.hstack([X_scaled, X_mh, X_tfidf])


# ── SVM prediction (from scratch) ───────────────────────────────────────────

def _svm_predict(X: np.ndarray) -> list[str]:
    """
    OvO SVM prediction using extracted support vectors and dual coefficients.

    For k classes, sklearn's OvO produces k*(k-1)/2 binary classifiers.
    dual_coef_ has shape (k-1, n_sv) — column j holds the coefficients for
    support vector j across all classifiers it participates in.
    """
    svm_p = _PARAMS['svm']
    classes = svm_p['classes']
    kernel = svm_p['kernel']
    intercept = np.array(svm_p['intercept'])
    n_support = svm_p['n_support']
    n_classes = len(classes)

    sv = _SUPPORT_VECTORS
    dual = _DUAL_COEF

    # Compute kernel matrix: K(X, support_vectors) -> (n_samples, n_sv)
    if kernel == 'linear':
        K = X @ sv.T
    elif kernel == 'rbf':
        gamma = svm_p['gamma_value']
        # ||x - sv||^2 = ||x||^2 - 2*x.sv^T + ||sv||^2
        X_sq = np.sum(X ** 2, axis=1, keepdims=True)       # (n, 1)
        sv_sq = np.sum(sv ** 2, axis=1, keepdims=True).T    # (1, n_sv)
        K = np.exp(-gamma * (X_sq - 2.0 * (X @ sv.T) + sv_sq))
    else:
        raise ValueError(f"Unsupported kernel: {kernel}")

    # OvO voting
    # For each pair (i, j) with i < j, compute decision function and vote
    # sklearn stores support vectors grouped by class: first n_support[0],
    # then n_support[1], etc.
    sv_offsets = np.cumsum([0] + n_support)

    predictions = []
    for sample_idx in range(X.shape[0]):
        votes = np.zeros(n_classes, dtype=int)
        k_row = K[sample_idx]  # (n_sv,)

        p = 0  # classifier index
        for i in range(n_classes):
            for j in range(i + 1, n_classes):
                # dual_coef for class i vs j:
                # - coefficients for class i's SVs are in dual_coef_[j-1] (row j-1)
                #   at columns sv_offsets[i]:sv_offsets[i+1]
                # - coefficients for class j's SVs are in dual_coef_[i] (row i)
                #   at columns sv_offsets[j]:sv_offsets[j+1]

                coef_i = dual[j - 1, sv_offsets[i]:sv_offsets[i + 1]]
                coef_j = dual[i, sv_offsets[j]:sv_offsets[j + 1]]

                k_i = k_row[sv_offsets[i]:sv_offsets[i + 1]]
                k_j = k_row[sv_offsets[j]:sv_offsets[j + 1]]

                decision = np.dot(coef_i, k_i) + np.dot(coef_j, k_j) + intercept[p]

                if decision > 0:
                    votes[i] += 1
                else:
                    votes[j] += 1
                p += 1

        predictions.append(classes[np.argmax(votes)])

    return predictions


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
    return _svm_predict(X)


if __name__ == '__main__':
    import sys
    fname = sys.argv[1] if len(sys.argv) > 1 else 'data/training_data_202601.csv'
    preds = predict_all(fname)
    print(f"Made {len(preds)} predictions")
    for name in sorted(set(preds)):
        count = preds.count(name)
        print(f"  {name}: {count}")

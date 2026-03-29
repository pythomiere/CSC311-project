"""
Shared preprocessing pipeline for all models.

Usage:
    from preprocess import (
        TARGET, NUMERIC_COLS, LIKERT_COLS, DOLLAR_COL,
        MULTI_SELECT_COLS, TEXT_COLS, SCALED_COLS,
        grouped_split, preprocess_split, build_features,
    )
"""

import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack as sp_hstack, csr_matrix

# ── Column names ──────────────────────────────────────────────────────────────

TARGET = 'Painting'

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


# ── Helpers ───────────────────────────────────────────────────────────────────

def extract_rating(response):
    """'4 - Agree' -> 4"""
    if pd.isna(response):
        return np.nan
    match = re.match(r'^(\d+)', str(response))
    return int(match.group(1)) if match else np.nan


def parse_dollar(val):
    """Extract numeric value from messy dollar strings."""
    if pd.isna(val):
        return np.nan
    s = str(val).lower().strip()
    s = re.sub(r'[\$,]', '', s)
    s = re.sub(r'(dollars|dollar|cad|cdn|ca|\bca\b)', '', s).strip()
    match = re.search(r'[\d]+\.?[\d]*', s)
    return float(match.group()) if match else np.nan


# ── Splitting ─────────────────────────────────────────────────────────────────

def grouped_split(df, seed=42, train_frac=0.70, val_frac=0.15):
    """Split by unique_id to prevent data leakage. Returns (df_train, df_val, df_test)."""
    unique_ids = df['unique_id'].unique()
    rng = np.random.RandomState(seed)
    rng.shuffle(unique_ids)

    n = len(unique_ids)
    n_train = int(train_frac * n)
    n_val = int(val_frac * n)

    train_ids = set(unique_ids[:n_train])
    val_ids = set(unique_ids[n_train:n_train + n_val])
    test_ids = set(unique_ids[n_train + n_val:])

    df_train = df[df['unique_id'].isin(train_ids)].copy()
    df_val = df[df['unique_id'].isin(val_ids)].copy()
    df_test = df[df['unique_id'].isin(test_ids)].copy()

    return df_train, df_val, df_test


# ── Per-split preprocessing ──────────────────────────────────────────────────

def preprocess_split(df_split, params, fit=False):
    """Apply Likert parsing, dollar parsing, numeric clipping/imputation.

    If fit=True, compute stats from df_split and populate params dict.
    Otherwise, use existing params.
    """
    df = df_split.copy()

    # Likert parsing
    for col in LIKERT_COLS:
        df[col] = df[col].apply(extract_rating)

    if fit:
        params['likert_modes'] = {col: float(df[col].mode()[0]) for col in LIKERT_COLS}
    for col in LIKERT_COLS:
        df[col] = df[col].fillna(params['likert_modes'][col])

    # Dollar parsing
    df['dollar_parsed'] = df[DOLLAR_COL].apply(parse_dollar)
    if fit:
        params['dollar_median'] = float(df['dollar_parsed'].median())
        params['dollar_p1'] = float(df['dollar_parsed'].quantile(0.01))
        params['dollar_p99'] = float(df['dollar_parsed'].quantile(0.99))
    df['dollar_parsed'] = df['dollar_parsed'].fillna(params['dollar_median'])
    df['dollar_parsed'] = df['dollar_parsed'].clip(params['dollar_p1'], params['dollar_p99'])
    df['dollar_log'] = np.log1p(df['dollar_parsed'])

    # Numeric clipping + imputation
    if fit:
        params['numeric_clip'] = {}
        params['numeric_medians'] = {}
    for col in NUMERIC_COLS:
        if fit:
            params['numeric_medians'][col] = float(df[col].median())
            params['numeric_clip'][col] = {
                'lo': float(df[col].quantile(0.01)),
                'hi': float(df[col].quantile(0.99)),
            }
        df[col] = df[col].fillna(params['numeric_medians'][col])
        df[col] = df[col].clip(
            params['numeric_clip'][col]['lo'],
            params['numeric_clip'][col]['hi'],
        )

    return df


# ── Multi-hot encoding ───────────────────────────────────────────────────────

def build_multi_hot_categories(df_train):
    """Build sorted category lists from training data."""
    categories = {}
    for col in MULTI_SELECT_COLS:
        cats = set()
        for val in df_train[col].dropna():
            for item in str(val).split(','):
                item = item.strip()
                if item:
                    cats.add(item)
        categories[col] = sorted(cats)
    return categories


def encode_multi_hot(df, categories_dict):
    """Encode multi-select columns as binary indicator arrays."""
    arrays = []
    for col, cats in categories_dict.items():
        mat = np.zeros((len(df), len(cats)), dtype=np.float64)
        for i, val in enumerate(df[col].values):
            if pd.isna(val):
                continue
            selected = {item.strip() for item in str(val).split(',')}
            for j, cat in enumerate(cats):
                if cat in selected:
                    mat[i, j] = 1.0
        arrays.append(mat)
    return np.hstack(arrays)


# ── Full feature building ────────────────────────────────────────────────────

def build_features(df_train, df_val, df_test, params, max_tfidf_features=200):
    """Run the full pipeline: preprocess, scale, multi-hot, TF-IDF.

    Returns (X_train, X_val, X_test, y_train, y_val, y_test) as sparse matrices,
    and populates params with all fitted parameters.
    """
    # Preprocessing (Likert, dollar, numeric)
    df_train = preprocess_split(df_train, params, fit=True)
    df_val = preprocess_split(df_val, params)
    df_test = preprocess_split(df_test, params)

    # Standardization
    scaler = StandardScaler()
    scaler.fit(df_train[SCALED_COLS])
    X_train_scaled = scaler.transform(df_train[SCALED_COLS])
    X_val_scaled = scaler.transform(df_val[SCALED_COLS])
    X_test_scaled = scaler.transform(df_test[SCALED_COLS])

    params['scaler_mean'] = scaler.mean_.tolist()
    params['scaler_scale'] = scaler.scale_.tolist()

    # Multi-hot
    multi_hot_cats = build_multi_hot_categories(df_train)
    params['multi_hot_categories'] = multi_hot_cats

    X_train_mh = encode_multi_hot(df_train, multi_hot_cats)
    X_val_mh = encode_multi_hot(df_val, multi_hot_cats)
    X_test_mh = encode_multi_hot(df_test, multi_hot_cats)

    # TF-IDF
    tfidf_vectorizers = {}
    X_train_tfidf_parts, X_val_tfidf_parts, X_test_tfidf_parts = [], [], []

    for col in TEXT_COLS:
        train_text = df_train[col].fillna('').astype(str)
        val_text = df_val[col].fillna('').astype(str)
        test_text = df_test[col].fillna('').astype(str)

        vec = TfidfVectorizer(
            max_features=max_tfidf_features,
            sublinear_tf=True,
            stop_words='english',
            min_df=2,
        )
        vec.fit(train_text)
        tfidf_vectorizers[col] = vec

        X_train_tfidf_parts.append(vec.transform(train_text))
        X_val_tfidf_parts.append(vec.transform(val_text))
        X_test_tfidf_parts.append(vec.transform(test_text))

    X_train_tfidf = sp_hstack(X_train_tfidf_parts)
    X_val_tfidf = sp_hstack(X_val_tfidf_parts)
    X_test_tfidf = sp_hstack(X_test_tfidf_parts)

    # Save TF-IDF params
    params['tfidf'] = {}
    for col, vec in tfidf_vectorizers.items():
        params['tfidf'][col] = {
            'vocabulary': {k: int(v) for k, v in vec.vocabulary_.items()},
            'idf': vec.idf_.tolist(),
        }

    # Combine all features
    X_train = sp_hstack([csr_matrix(X_train_scaled), csr_matrix(X_train_mh), X_train_tfidf])
    X_val = sp_hstack([csr_matrix(X_val_scaled), csr_matrix(X_val_mh), X_val_tfidf])
    X_test = sp_hstack([csr_matrix(X_test_scaled), csr_matrix(X_test_mh), X_test_tfidf])

    y_train = df_train[TARGET].values
    y_val = df_val[TARGET].values
    y_test = df_test[TARGET].values

    return X_train, X_val, X_test, y_train, y_val, y_test

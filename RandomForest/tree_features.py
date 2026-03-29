import numpy as np
import pandas as pd
import re

TARGET_COL = "Painting"
GROUP_COL = "unique_id"

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

CATEGORICAL_COLS = {
    "room": "If you could purchase this painting, which room would you put that painting in?",
    "companion": "If you could view this art in person, who would you want to view it with?",
    "season": "What season does this art piece remind you of?",
}

TEXT_COLS = [
  "Describe how this painting makes you feel.",
  "If this painting was a food, what would be?",
  "Imagine a soundtrack for this painting. Describe that soundtrack without naming any objects in the painting.",
]
MONEY_COL = "How much (in Canadian dollars) would you be willing to pay for this painting?"


def parse_likert(x):
    if pd.isna(x):
        return np.nan

    x = str(x).strip()
    if not x:
        return np.nan

    first_char = x[0]
    if first_char in {'1', '2', '3', '4', '5'}:
        return int(first_char)

    return np.nan

def split_multi_select(x):
    if pd.isna(x):
        return []

    x = str(x).strip()
    if not x:
        return []

    parts = x.split(',')
    cleaned = []

    for part in parts:
        item = part.strip()
        if item:
            cleaned.append(item)

    return cleaned

def parse_float_or_nan(x):
    if pd.isna(x):
        return np.nan
    try: 
        return float(x)
    except (TypeError, ValueError):
        return np.nan

def parse_money(x):
    if pd.isna(x):
        return np.nan
    text = str(x).strip().lower()
    if not text:
        return np.nan
    text = text.replace("\u00a0", " ")
    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.replace("cad", "")
    text = text.replace("canadian dollars", "")
    text = text.replace("canadian dollar", "")
    text = text.replace("dollars", "")
    text = text.replace("dollar", "")
    text = text.strip()

    text = re.sub(r'(?<=\d) (?=\d)', '', text)
    match = re.search(r"(\d+(?:\.\d+)?)", text)
    if match is None:
      return np.nan

    value = float(match.group(1))

    if re.search(r"\bbillion\b|\bbn\b|\bb\b", text):
      value *= 1_000_000_000
    elif re.search(r"\bmillion\b|\bm\b", text):
      value *= 1_000_000
    elif re.search(r"\bthousand\b|\bk\b", text):
      value *= 1_000

    return value


def _transform_numeric_series(series, col, clip_bounds):
    if col == MONEY_COL:
        values = series.apply(parse_money)
    else:
        values = series.apply(parse_float_or_nan)

    lower, upper = clip_bounds
    values = values.clip(lower=lower, upper=upper)

    if col == MONEY_COL:
        values = values.apply(lambda x: np.log1p(max(x, 0)) if pd.notna(x) else np.nan)

    return values

def fit_tree_preprocessor(df_train):
    state = {}

    state["numeric_medians"] = {}
    state["clip_bounds"] = {}
    state["likert_modes"] = {}
    state["categorical_vocabularies"] = {}

    # numeric columns
    for col in NUMERIC_COLS:
      if col == MONEY_COL:
          parsed_values = df_train[col].apply(parse_money)
      else:
          parsed_values = df_train[col].apply(parse_float_or_nan)

      state["clip_bounds"][col] = (
          parsed_values.quantile(0.01),
          parsed_values.quantile(0.99),
      )
      transformed_values = _transform_numeric_series(df_train[col], col, state["clip_bounds"][col])
      state["numeric_medians"][col] = transformed_values.median()

    # likert columns
    for col in LIKERT_COLS:
      values = df_train[col].apply(parse_likert)
      state["likert_modes"][col] = values.mode().iloc[0]

    # categorical vocabularies
    for key, col in CATEGORICAL_COLS.items():
      vocab = set()

      for x in df_train[col]:
          items = split_multi_select(x)
          for item in items:
              vocab.add(item)

      state["categorical_vocabularies"][key] = sorted(vocab)

    feature_names = []
    feature_names.extend(NUMERIC_COLS)
    feature_names.extend(LIKERT_COLS)

    for key in CATEGORICAL_COLS:
      for category in state["categorical_vocabularies"][key]:
        feature_names.append(f"{key}__{category}")

    for col in NUMERIC_COLS:
      feature_names.append(f"missing__{col}")

    for col in LIKERT_COLS:
      feature_names.append(f"missing__{col}")

    for key in CATEGORICAL_COLS:
      feature_names.append(f"missing__{key}")

    state["feature_names"] = feature_names

    return state


def transform_tree_features(df, state):
    feature_dict = {}

    for col in NUMERIC_COLS:
      values = _transform_numeric_series(df[col], col, state["clip_bounds"][col])
      values = values.fillna(state["numeric_medians"][col])
      feature_dict[col] = values

    for col in LIKERT_COLS:
      values = df[col].apply(parse_likert)
      values = values.fillna(state["likert_modes"][col])
      feature_dict[col] = values

    for key, col in CATEGORICAL_COLS.items():
      split_values = df[col].apply(split_multi_select)

      for category in state["categorical_vocabularies"][key]:
        feature_name = f"{key}__{category}"
        feature_dict[feature_name] = split_values.apply(
            lambda items: 1 if category in items else 0
        )

    for col in NUMERIC_COLS:
      if col == MONEY_COL:
        missing = df[col].apply(parse_money).isna().astype(int)
      else:
        missing = df[col].apply(parse_float_or_nan).isna().astype(int)
      feature_dict[f"missing__{col}"] = missing

    for col in LIKERT_COLS:
      feature_dict[f"missing__{col}"] = df[col].apply(parse_likert).isna().astype(int)

    for key, col in CATEGORICAL_COLS.items():
      feature_dict[f"missing__{key}"] = df[col].isna().astype(int)

    X = pd.DataFrame(feature_dict)
    X = X[state["feature_names"]]

    if X.isna().sum().sum() != 0:
      raise ValueError("transform_tree_features produced NaNs")

    return X

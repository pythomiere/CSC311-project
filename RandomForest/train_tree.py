import json
import pickle
from itertools import product
from pathlib import Path

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report

try:
    from tree_features import fit_tree_preprocessor, transform_tree_features, TARGET_COL, GROUP_COL
except ModuleNotFoundError:
    from src.tree_features import fit_tree_preprocessor, transform_tree_features, TARGET_COL, GROUP_COL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "training_data_202601.csv"
SPLIT_PATH = PROJECT_ROOT / "artifacts" / "splits" / "group_split_seed42.json"
RESULTS_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "tree_grid_results.csv"
MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "best_tree_val_model.pkl"
STATE_PATH = PROJECT_ROOT / "artifacts" / "preprocessor" / "tree_val_preprocessor_state.pkl"
RANDOM_STATE = 42


def load_split_artifact(split_path):
    with open(split_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_pickle_artifact(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def build_split_dataframes(df, split):
    train_ids = split["train_ids"]
    val_ids = split["val_ids"]
    test_ids = split["test_ids"]

    train_df = df[df[GROUP_COL].isin(train_ids)].copy()
    val_df = df[df[GROUP_COL].isin(val_ids)].copy()
    test_df = df[df[GROUP_COL].isin(test_ids)].copy()

    return train_df, val_df, test_df


def build_features_and_labels(train_df, val_df):
    state = fit_tree_preprocessor(train_df)

    X_train = transform_tree_features(train_df, state)
    X_val = transform_tree_features(val_df, state)

    y_train = train_df[TARGET_COL]
    y_val = val_df[TARGET_COL]

    return state, X_train, X_val, y_train, y_val


def train_tree(X_train, y_train, **kwargs):
    model = DecisionTreeClassifier(random_state=RANDOM_STATE, **kwargs)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_train, y_train, X_val, y_val, print_details=True):
    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_acc = accuracy_score(y_train, train_pred)
    val_acc = accuracy_score(y_val, val_pred)
    val_macro_f1 = f1_score(y_val, val_pred, average="macro")
    cm = confusion_matrix(y_val, val_pred, labels=model.classes_)
    report = classification_report(y_val, val_pred, zero_division=0)

    if print_details:
        print(f"Train accuracy: {train_acc:.4f}")
        print(f"Validation accuracy: {val_acc:.4f}")
        print(f"Validation macro-F1: {val_macro_f1:.4f}")
        print("\nConfusion matrix:")
        print(cm)
        print("\nClassification report:")
        print(report)

    metrics = {
        "train_accuracy": train_acc,
        "val_accuracy": val_acc,
        "val_macro_f1": val_macro_f1,
        "confusion_matrix": cm,
        "classes": list(model.classes_),
        "classification_report": report,
    }
    return metrics


def run_tree_grid_search(X_train, y_train, X_val, y_val):
    grid = {
        "criterion": ["gini", "entropy"],
        "max_depth": [3, 5, 8, 12, None],
        "min_samples_leaf": [1, 2, 5, 10],
        "ccp_alpha": [0.0, 1e-4, 1e-3, 1e-2],
        "class_weight": [None, "balanced"],
    }

    results = []

    for criterion, max_depth, min_samples_leaf, ccp_alpha, class_weight in product(
        grid["criterion"],
        grid["max_depth"],
        grid["min_samples_leaf"],
        grid["ccp_alpha"],
        grid["class_weight"],
    ):
        model = train_tree(
            X_train,
            y_train,
            criterion=criterion,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            ccp_alpha=ccp_alpha,
            class_weight=class_weight,
        )

        metrics = evaluate_model(model, X_train, y_train, X_val, y_val, print_details=False)
        results.append(
            {
                "criterion": criterion,
                "max_depth": max_depth,
                "min_samples_leaf": min_samples_leaf,
                "ccp_alpha": ccp_alpha,
                "class_weight": class_weight,
                "train_accuracy": metrics["train_accuracy"],
                "val_accuracy": metrics["val_accuracy"],
                "val_macro_f1": metrics["val_macro_f1"],
            }
        )

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by=["val_accuracy", "val_macro_f1", "train_accuracy"],
        ascending=[False, False, True],
    ).reset_index(drop=True)

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(RESULTS_PATH, index=False)
    return results_df


def main():
    df = pd.read_csv(DATA_PATH)
    split = load_split_artifact(SPLIT_PATH)

    train_df, val_df, test_df = build_split_dataframes(df, split)

    state, X_train, X_val, y_train, y_val = build_features_and_labels(train_df, val_df)

    print("Rows:")
    print("train:", len(train_df))
    print("val:", len(val_df))
    print("test:", len(test_df))
    print("features:", X_train.shape[1])

    print("\nBaseline tree")
    baseline_model = train_tree(X_train, y_train)
    evaluate_model(baseline_model, X_train, y_train, X_val, y_val)

    print("\nRunning grid search...")
    results_df = run_tree_grid_search(X_train, y_train, X_val, y_val)

    print("\nTop 10 grid-search results:")
    print(results_df.head(10).to_string(index=False))

    best_row = results_df.iloc[0]
    print("\nBest hyperparameters:")
    print(best_row.to_dict())

    print("\nBest tuned tree")
    best_model = train_tree(
        X_train,
        y_train,
        criterion=best_row["criterion"],
        max_depth=None if pd.isna(best_row["max_depth"]) else int(best_row["max_depth"]),
        min_samples_leaf=int(best_row["min_samples_leaf"]),
        ccp_alpha=float(best_row["ccp_alpha"]),
        class_weight=None if pd.isna(best_row["class_weight"]) else best_row["class_weight"],
    )
    evaluate_model(best_model, X_train, y_train, X_val, y_val)

    save_pickle_artifact(best_model, MODEL_PATH)
    save_pickle_artifact(state, STATE_PATH)


if __name__ == "__main__":
    main()

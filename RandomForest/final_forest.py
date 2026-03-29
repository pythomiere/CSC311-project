import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

try:
    from train_forest import (
        build_split_dataframes,
        load_split_artifact,
        save_pickle_artifact,
        train_forest,
    )
    from tree_features import TARGET_COL, fit_tree_preprocessor, transform_tree_features
except ModuleNotFoundError:
    from src.train_forest import (
        build_split_dataframes,
        load_split_artifact,
        save_pickle_artifact,
        train_forest,
    )
    from src.tree_features import TARGET_COL, fit_tree_preprocessor, transform_tree_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "training_data_202601.csv"
SPLIT_PATH = PROJECT_ROOT / "artifacts" / "splits" / "group_split_seed42.json"

METRICS_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "forest_test_metrics.json"
REPORT_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "forest_test_classification_report.txt"
CM_PATH = PROJECT_ROOT / "artifacts" / "metrics" / "forest_test_confusion_matrix.csv"

MODEL_PATH = PROJECT_ROOT / "artifacts" / "models" / "final_forest_trainval.pkl"
STATE_PATH = PROJECT_ROOT / "artifacts" / "preprocessor" / "final_forest_trainval_state.pkl"

BEST_PARAMS = {
    "n_estimators": 300,
    "criterion": "gini",
    "max_depth": None,
    "max_features": 0.5,
    "min_samples_leaf": 1,
    "class_weight": None,
}


def save_json_artifact(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_text_artifact(text, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    df = pd.read_csv(DATA_PATH)
    split = load_split_artifact(SPLIT_PATH)
    label_order = split["label_order"]

    train_df, val_df, test_df = build_split_dataframes(df, split)
    train_val_df = pd.concat([train_df, val_df], ignore_index=True)

    state = fit_tree_preprocessor(train_val_df)

    X_train_val = transform_tree_features(train_val_df, state)
    X_test = transform_tree_features(test_df, state)

    y_train_val = train_val_df[TARGET_COL]
    y_test = test_df[TARGET_COL]

    model = train_forest(X_train_val, y_train_val, **BEST_PARAMS)
    test_pred = model.predict(X_test)

    test_acc = accuracy_score(y_test, test_pred)
    test_macro_f1 = f1_score(y_test, test_pred, average="macro")
    cm = confusion_matrix(y_test, test_pred, labels=label_order)
    report = classification_report(
        y_test,
        test_pred,
        labels=label_order,
        target_names=label_order,
        zero_division=0,
    )

    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test macro-F1: {test_macro_f1:.4f}")
    print("\nConfusion matrix:")
    print(cm)
    print("\nClassification report:")
    print(report)

    cm_df = pd.DataFrame(cm, index=label_order, columns=label_order)
    CM_PATH.parent.mkdir(parents=True, exist_ok=True)
    cm_df.to_csv(CM_PATH)

    metrics = {
        "best_params": BEST_PARAMS,
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_val_rows": len(train_val_df),
        "test_rows": len(test_df),
        "feature_count": int(X_train_val.shape[1]),
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_macro_f1),
        "label_order": label_order,
    }

    save_json_artifact(metrics, METRICS_PATH)
    save_text_artifact(report, REPORT_PATH)
    save_pickle_artifact(model, MODEL_PATH)
    save_pickle_artifact(state, STATE_PATH)


if __name__ == "__main__":
    main()

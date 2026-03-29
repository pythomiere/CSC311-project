import json
import pickle
from pathlib import Path

import numpy as np

try:
    from tree_features import CATEGORICAL_COLS, LIKERT_COLS, MONEY_COL, NUMERIC_COLS, TARGET_COL
except ModuleNotFoundError:
    from src.tree_features import CATEGORICAL_COLS, LIKERT_COLS, MONEY_COL, NUMERIC_COLS, TARGET_COL


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_IN = PROJECT_ROOT / "artifacts" / "models" / "final_forest_trainval.pkl"
STATE_IN = PROJECT_ROOT / "artifacts" / "preprocessor" / "final_forest_trainval_state.pkl"

MODEL_OUT = PROJECT_ROOT / "forest_model.npz"
META_OUT = PROJECT_ROOT / "forest_meta.json"


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main():
    with open(MODEL_IN, "rb") as f:
        model = pickle.load(f)

    with open(STATE_IN, "rb") as f:
        state = pickle.load(f)

    arrays = {}
    for tree_idx, estimator in enumerate(model.estimators_):
        tree = estimator.tree_
        prefix = f"tree_{tree_idx}"

        arrays[f"{prefix}_children_left"] = tree.children_left.astype(np.int32)
        arrays[f"{prefix}_children_right"] = tree.children_right.astype(np.int32)
        arrays[f"{prefix}_feature"] = tree.feature.astype(np.int32)
        arrays[f"{prefix}_threshold"] = tree.threshold.astype(np.float32)
        arrays[f"{prefix}_value"] = tree.value[:, 0, :].astype(np.float32)

    np.savez_compressed(MODEL_OUT, **arrays)

    state_export = {
        "target_col": TARGET_COL,
        "numeric_cols": NUMERIC_COLS,
        "likert_cols": LIKERT_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "money_col": MONEY_COL,
        **to_jsonable(state),
    }

    meta = {
        "classes": [str(label) for label in model.classes_],
        "tree_count": len(model.estimators_),
        "state": state_export,
    }

    with open(META_OUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote {MODEL_OUT}")
    print(f"Wrote {META_OUT}")


if __name__ == "__main__":
    main()

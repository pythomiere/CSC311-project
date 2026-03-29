"""
Train logistic regression and export weights for standalone prediction.

Outputs: lr_model_params.json (preprocessing params + LR weights)

Usage: python train_lr.py
"""

import json
import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from preprocess import grouped_split, build_features


def main():
    df = pd.read_csv('data/training_data_202601.csv')
    print(f"Loaded {len(df)} rows")

    # ── Grouped split (70/15/15 by student) ──────────────────────────────────
    df_train, df_val, df_test = grouped_split(df)
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # ── Build features ───────────────────────────────────────────────────────
    params = {}
    X_train, X_val, X_test, y_train, y_val, y_test = build_features(
        df_train, df_val, df_test, params
    )
    print(f"Feature dimension: {X_train.shape[1]}")

    # ── Hyperparameter tuning ────────────────────────────────────────────────
    # Grid search over C and penalty type as described in the report
    print("\nRunning hyperparameter grid search...")
    best_val_acc = 0
    best_cfg = {}
    best_model = None

    for penalty in ['l1', 'l2']:
        solver = 'saga' if penalty == 'l1' else 'lbfgs'
        for C in [0.01, 0.1, 1, 10, 100]:
            clf = LogisticRegression(
                solver=solver,
                penalty=penalty,
                C=C,
                max_iter=2000,
            )
            clf.fit(X_train, y_train)
            val_acc = accuracy_score(y_val, clf.predict(X_val))
            marker = ''
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_cfg = {'penalty': penalty, 'C': C, 'solver': solver}
                best_model = clf
                marker = ' <-- best'
            print(f"  penalty={penalty}, C={C:>6} -> val_acc={val_acc:.4f}{marker}")

    print(f"\nBest: {best_cfg}, val_acc={best_val_acc:.4f}")

    # ── Final evaluation ─────────────────────────────────────────────────────
    train_acc = accuracy_score(y_train, best_model.predict(X_train))
    test_acc = accuracy_score(y_test, best_model.predict(X_test))
    print(f"Train acc: {train_acc:.4f}")
    print(f"Val acc:   {best_val_acc:.4f}")
    print(f"Test acc:  {test_acc:.4f}")
    print("\nTest classification report:")
    print(classification_report(y_test, best_model.predict(X_test)))

    # ── Export model ─────────────────────────────────────────────────────────
    params['lr'] = {
        'classes': best_model.classes_.tolist(),
        'coef': best_model.coef_.tolist(),
        'intercept': best_model.intercept_.tolist(),
        'best_params': best_cfg,
    }

    with open('lr_model_params.json', 'w') as f:
        json.dump(params, f)

    size = os.path.getsize('lr_model_params.json') / 1024
    print(f"\nSaved: lr_model_params.json ({size:.1f} KB)")


if __name__ == '__main__':
    main()

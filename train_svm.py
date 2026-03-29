"""
Train the SVM model and export all parameters needed for standalone prediction.

Outputs: svm_model_params.json, svm_model_arrays.npz

Usage: python train_svm.py
"""

import json
import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

from preprocess import TARGET, grouped_split, build_features


def main():
    df = pd.read_csv('data/training_data_202601.csv')
    print(f"Loaded {len(df)} rows")

    # ── Grouped split ────────────────────────────────────────────────────────
    df_train, df_val, df_test = grouped_split(df)
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # ── Build features ───────────────────────────────────────────────────────
    params = {}
    X_train, X_val, X_test, y_train, y_val, y_test = build_features(
        df_train, df_val, df_test, params
    )
    print(f"Feature dimension: {X_train.shape[1]}")

    # ── Grid search ──────────────────────────────────────────────────────────
    print("\nRunning hyperparameter grid search...")
    best_val_acc = 0
    best_params_svm = {}
    best_model = None

    configs = []
    for C in [0.1, 1, 10, 100]:
        configs.append({'kernel': 'linear', 'C': C, 'decision_function_shape': 'ovo'})
        for gamma in ['scale', 'auto', 0.1, 0.01]:
            configs.append({
                'kernel': 'rbf', 'C': C, 'gamma': gamma,
                'decision_function_shape': 'ovo',
            })

    for i, cfg in enumerate(configs):
        svm = SVC(**cfg)
        svm.fit(X_train, y_train)
        val_acc = accuracy_score(y_val, svm.predict(X_val))
        marker = ''
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_params_svm = cfg
            best_model = svm
            marker = ' <-- best'
        print(f"  [{i+1}/{len(configs)}] {cfg} -> val_acc={val_acc:.4f}{marker}")

    print(f"\nBest: {best_params_svm}, val_acc={best_val_acc:.4f}")

    # ── Final evaluation ─────────────────────────────────────────────────────
    train_acc = accuracy_score(y_train, best_model.predict(X_train))
    test_acc = accuracy_score(y_test, best_model.predict(X_test))
    print(f"Train acc: {train_acc:.4f}")
    print(f"Val acc:   {best_val_acc:.4f}")
    print(f"Test acc:  {test_acc:.4f}")
    print("\nTest classification report:")
    print(classification_report(y_test, best_model.predict(X_test)))

    # ── Export model ─────────────────────────────────────────────────────────
    params['svm'] = {
        'kernel': best_model.kernel,
        'gamma_value': float(best_model._gamma) if best_model.kernel == 'rbf' else None,
        'decision_function_shape': best_model.decision_function_shape,
        'classes': best_model.classes_.tolist(),
        'intercept': best_model.intercept_.tolist(),
        'n_support': best_model.n_support_.tolist(),
    }

    np.savez_compressed(
        'svm_model_arrays.npz',
        support_vectors=best_model.support_vectors_.toarray()
            if hasattr(best_model.support_vectors_, 'toarray')
            else best_model.support_vectors_,
        dual_coef=best_model.dual_coef_.toarray()
            if hasattr(best_model.dual_coef_, 'toarray')
            else best_model.dual_coef_,
    )

    with open('svm_model_params.json', 'w') as f:
        json.dump(params, f)

    json_size = os.path.getsize('svm_model_params.json') / 1024
    npz_size = os.path.getsize('svm_model_arrays.npz') / 1024
    print(f"\nSaved: svm_model_params.json ({json_size:.1f} KB)")
    print(f"Saved: svm_model_arrays.npz ({npz_size:.1f} KB)")
    print(f"Total: {(json_size + npz_size):.1f} KB")


if __name__ == '__main__':
    main()

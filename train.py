"""Train and compare churn models, then save the best one."""
import json

import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn_utils import (CATEGORICAL_FEATURES, FEATURES, METRICS_PATH,
                         MODEL_PATH, NUMERIC_FEATURES, TARGET, load_data)

RANDOM_STATE = 42


def build_pipeline(model) -> Pipeline:
    preprocess = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])
    return Pipeline([("preprocess", preprocess), ("model", model)])


def evaluate(pipe, X_test, y_test, threshold=0.5) -> dict:
    proba = pipe.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
    }


def main():
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=400, min_samples_leaf=5, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results, fitted = {}, {}
    for name, model in candidates.items():
        pipe = build_pipeline(model).fit(X_train, y_train)
        results[name] = evaluate(pipe, X_test, y_test)
        fitted[name] = pipe
        print(f"{name:20s} ROC-AUC={results[name]['roc_auc']:.3f} "
              f"Recall={results[name]['recall']:.3f} F1={results[name]['f1']:.3f}")

    # F1 balances catching churners (recall) against false alarms (precision),
    # which matters more for a retention team than raw accuracy.
    best_name = max(results, key=lambda n: results[n]["f1"])
    best = fitted[best_name]

    # Feature importance in terms of the original (pre-encoding) columns.
    names = best.named_steps["preprocess"].get_feature_names_out()
    model = best.named_steps["model"]
    if hasattr(model, "coef_"):
        weights = abs(model.coef_[0])
    else:
        weights = model.feature_importances_
    importance = {}
    for raw_name, w in zip(names, weights):
        col = raw_name.split("__", 1)[1]
        base = next((f for f in FEATURES if col == f or col.startswith(f + "_")), col)
        importance[base] = importance.get(base, 0.0) + float(w)
    total = sum(importance.values())
    importance = dict(sorted(((k, v / total) for k, v in importance.items()),
                             key=lambda kv: kv[1], reverse=True))

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(best, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps({
        "best_model": best_name,
        "comparison": results,
        "feature_importance": importance,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }, indent=2))
    print(f"\nSaved best model: {best_name}")


if __name__ == "__main__":
    main()

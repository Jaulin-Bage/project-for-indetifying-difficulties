import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

import util
from params import RESULT_VECTOR_MAPPING
from selfOptimization import encode

LABEL_COLUMN = "实际认定结果"
DEFAULT_TRAIN_DATASET = Path("true_data") / "三所学校一致数据_基于三个指标体系.csv"
DEFAULT_TEST_DATASET = Path("true_data") / "total_20260613.csv"
DEFAULT_MIN_IMPORTANCE_PERCENT = 2.0
DEFAULT_PRIOR_STRENGTH = 0.3


def build_feature_frame(dataset_path, indicator_path):
    df_zbtx = pd.read_csv(indicator_path)
    _, a_ii = util.get_AI_AII(df_zbtx)

    a_ii_labels = []
    for ai in a_ii.keys():
        for aii in a_ii[ai].keys():
            a_ii_labels.append(aii)

    encoded_rows = encode(dataset_path, df_zbtx=df_zbtx, with_RRDA=False)
    for i in range(len(encoded_rows)):
        encoded_rows[i] = {
            key: value
            for key, value in sorted(
                encoded_rows[i].items(),
                key=lambda item: a_ii_labels.index(item[0]),
            )
        }

    x = pd.DataFrame(encoded_rows)
    raw_df = pd.read_csv(dataset_path)
    if LABEL_COLUMN not in raw_df.columns:
        raise ValueError(f"Dataset missing label column: {LABEL_COLUMN}")

    y = raw_df[LABEL_COLUMN].map(RESULT_VECTOR_MAPPING)
    invalid_labels = raw_df.loc[y.isna(), LABEL_COLUMN].dropna().unique()
    real_invalid_labels = [
        label for label in invalid_labels
        if str(label).strip() and str(label).strip() != LABEL_COLUMN
    ]
    if real_invalid_labels:
        raise ValueError(f"Unknown label values: {real_invalid_labels}")

    valid_mask = y.notna()
    dropped_rows = len(y) - int(valid_mask.sum())
    if dropped_rows:
        print(f"Dropped rows without valid labels: {dropped_rows}")

    return x.loc[valid_mask].reset_index(drop=True), y.loc[valid_mask].astype(int), a_ii_labels


def make_model(model_name, random_state):
    if model_name == "tree":
        return DecisionTreeClassifier(
            criterion="gini",
            max_depth=5,
            min_samples_leaf=5,
            random_state=random_state,
        )
    if model_name == "forest":
        return RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=400,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )
    if model_name == "logistic":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                C=0.5,
                class_weight="balanced",
                max_iter=5000,
                random_state=random_state,
            )),
        ])
    if model_name == "linear_svm":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LinearSVC(
                C=0.5,
                class_weight="balanced",
                dual=False,
                max_iter=5000,
                random_state=random_state,
            )),
        ])
    if model_name == "knn":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(
                n_neighbors=7,
                weights="distance",
            )),
        ])
    if model_name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=3,
            random_state=random_state,
        )
    if model_name == "bayes":
        return GaussianNB()
    raise ValueError(f"Unsupported model: {model_name}")


def normalize_importances(importances, feature_count):
    importances = np.maximum(np.asarray(importances, dtype=float), 0)
    total = float(np.sum(importances))
    if total <= 0:
        return np.ones(feature_count, dtype=float) / feature_count
    return importances / total


def load_prior_importances(indicator_path, feature_names):
    indicator_df = pd.read_csv(indicator_path)
    if "indicator_2" not in indicator_df.columns or "score_max" not in indicator_df.columns:
        return np.ones(len(feature_names), dtype=float) / len(feature_names)

    prior_map = (
        indicator_df.dropna(subset=["indicator_2"])
        .drop_duplicates("indicator_2", keep="first")
        .set_index("indicator_2")["score_max"]
        .to_dict()
    )
    prior = np.array(
        [pd.to_numeric(prior_map.get(name, 0), errors="coerce") for name in feature_names],
        dtype=float,
    )
    prior = np.nan_to_num(prior, nan=0.0, posinf=0.0, neginf=0.0)
    return normalize_importances(prior, len(feature_names))


def apply_importance_constraints(
    importances,
    feature_names,
    indicator_path,
    min_importance_percent=DEFAULT_MIN_IMPORTANCE_PERCENT,
    prior_strength=DEFAULT_PRIOR_STRENGTH,
):
    model_weights = normalize_importances(importances, len(feature_names))

    prior_strength = min(max(float(prior_strength), 0.0), 1.0)
    if prior_strength > 0:
        prior_weights = load_prior_importances(indicator_path, feature_names)
        constrained = (1.0 - prior_strength) * model_weights + prior_strength * prior_weights
    else:
        constrained = model_weights

    min_weight = max(float(min_importance_percent), 0.0) / 100.0
    if min_weight * len(feature_names) >= 1.0:
        raise ValueError("单项最低权重过大，指标数量较多时所有最低权重之和不能达到或超过 100%。")

    constrained = normalize_importances(constrained, len(feature_names))
    remaining_weight = 1.0 - min_weight * len(feature_names)
    return min_weight + remaining_weight * constrained


def get_model_importances(model):
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.named_steps.get("classifier", model.steps[-1][1])

    if hasattr(estimator, "feature_importances_"):
        return estimator.feature_importances_

    if hasattr(estimator, "coef_"):
        coefficients = np.asarray(estimator.coef_, dtype=float)
        if coefficients.ndim == 1:
            return np.abs(coefficients)
        return np.mean(np.abs(coefficients), axis=0)

    return None


def save_feature_importance(
    model,
    feature_names,
    output_dir,
    model_name,
    x_test,
    y_test,
    random_state,
    indicator_path,
    min_importance_percent=DEFAULT_MIN_IMPORTANCE_PERCENT,
    prior_strength=DEFAULT_PRIOR_STRENGTH,
):
    importances = get_model_importances(model)
    if importances is None:
        result = permutation_importance(
            model,
            x_test,
            y_test,
            n_repeats=10,
            random_state=random_state,
            scoring="accuracy",
            n_jobs=-1,
        )
        importances = np.maximum(result.importances_mean, 0)

    importances = apply_importance_constraints(
        importances,
        feature_names,
        indicator_path,
        min_importance_percent=min_importance_percent,
        prior_strength=prior_strength,
    )

    importance_df = pd.DataFrame({
        "indicator": feature_names,
        "importance": importances * 100,
    }).sort_values("importance", ascending=False)

    output_path = output_dir / f"traditional_{model_name}_feature_importance.csv"
    importance_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def generate_indicator_system(indicator_path, importance_path, output_path):
    indicator_df = pd.read_csv(indicator_path)
    importance_df = pd.read_csv(importance_path)
    weight_map = importance_df.set_index("indicator")["importance"].to_dict()

    output_df = indicator_df.copy()
    output_df["score_max"] = output_df["indicator_2"].map(weight_map).fillna(
        output_df["score_max"]
    )
    output_df["score"] = output_df["normalized_score"] * output_df["score_max"]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path, output_df


def train_traditional_model(
    dataset_path,
    indicator_path,
    output_dir,
    test_dataset_path=DEFAULT_TEST_DATASET,
    model_name="forest",
    random_state=2024,
    new_indicator_path=None,
    min_importance_percent=DEFAULT_MIN_IMPORTANCE_PERCENT,
    prior_strength=DEFAULT_PRIOR_STRENGTH,
):
    dataset_path = Path(dataset_path)
    indicator_path = Path(indicator_path)
    output_dir = Path(output_dir)
    test_dataset_path = Path(test_dataset_path)

    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, feature_names = build_feature_frame(dataset_path, indicator_path)
    x_train, y_train = x, y
    x_test, y_test, _ = build_feature_frame(test_dataset_path, indicator_path)
    x_test = x_test.reindex(columns=feature_names, fill_value=0)

    model = make_model(model_name, random_state)
    if model_name == "knn":
        model.set_params(classifier__n_neighbors=max(1, min(7, len(x_train))))
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    class_labels = sorted(set(y_train) | set(y_test) | set(test_pred))
    matrix = confusion_matrix(y_test, test_pred, labels=class_labels)

    model_path = output_dir / f"traditional_{model_name}.joblib"
    joblib.dump(model, model_path)
    importance_path = save_feature_importance(
        model,
        feature_names,
        output_dir,
        model_name,
        x_test,
        y_test,
        random_state,
        indicator_path,
        min_importance_percent=min_importance_percent,
        prior_strength=prior_strength,
    )
    if new_indicator_path is None:
        new_indicator_path = output_dir / f"traditional_{model_name}_indicator_system.csv"
    new_indicator_path, new_indicator_df = generate_indicator_system(
        indicator_path, importance_path, new_indicator_path
    )

    report_path = output_dir / f"traditional_{model_name}_report.txt"
    report_path.write_text(
        "\n".join([
            f"Model: {model_name}",
            f"Train samples: {len(x_train)}, test samples: {len(x_test)}",
            f"Train accuracy: {train_acc:.4f}",
            f"Test accuracy: {test_acc:.4f}",
            f"Minimum importance percent: {min_importance_percent:.4f}",
            f"Prior strength: {prior_strength:.4f}",
        ]),
        encoding="utf-8",
    )

    matrix_path = output_dir / f"traditional_{model_name}_confusion_matrix.csv"
    pd.DataFrame(matrix).to_csv(matrix_path, index=False, encoding="utf-8-sig")

    print(f"Model: {model_name}")
    print(f"Train samples: {len(x_train)}, test samples: {len(x_test)}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("Confusion matrix:")
    print(matrix)
    print(f"\nSaved model: {model_path}")
    print(f"Saved report: {report_path}")
    print(f"Saved confusion matrix: {matrix_path}")
    if importance_path is not None:
        print(f"Saved feature importance: {importance_path}")

    return {
        "model": model,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_samples": len(x_train),
        "test_samples": len(x_test),
        "confusion_matrix": matrix,
        "class_labels": class_labels,
        "feature_names": feature_names,
        "model_path": model_path,
        "report_path": report_path,
        "matrix_path": matrix_path,
        "importance_path": importance_path,
        "new_indicator_path": new_indicator_path,
        "new_indicator_df": new_indicator_df,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train a traditional ML baseline.")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_TRAIN_DATASET,
        help="CSV training dataset with actual labels.",
    )
    parser.add_argument("--train_dataset", help="CSV training dataset with actual labels.")
    parser.add_argument(
        "--test_dataset",
        default=DEFAULT_TEST_DATASET,
        help="CSV testing dataset with actual labels.",
    )
    parser.add_argument("--old_zbtx_file", required=True, help="Indicator system CSV.")
    parser.add_argument("--output_dir", default="tmp", help="Directory for model outputs.")
    parser.add_argument(
        "--model",
        choices=[
            "tree",
            "forest",
            "extra_trees",
            "logistic",
            "linear_svm",
            "knn",
            "gradient_boosting",
            "bayes",
        ],
        default="forest",
        help="Traditional model to train.",
    )
    parser.add_argument("--random_state", type=int, default=2024)
    parser.add_argument(
        "--min_importance_percent",
        type=float,
        default=DEFAULT_MIN_IMPORTANCE_PERCENT,
        help="Minimum percent reserved for every indicator after training.",
    )
    parser.add_argument(
        "--prior_strength",
        type=float,
        default=DEFAULT_PRIOR_STRENGTH,
        help="Blend ratio for the original indicator score_max prior, from 0 to 1.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    dataset_path = args.train_dataset or args.dataset
    if dataset_path is None:
        raise ValueError("Please provide --dataset or --train_dataset.")

    train_traditional_model(
        dataset_path=dataset_path,
        indicator_path=args.old_zbtx_file,
        output_dir=Path(args.output_dir),
        test_dataset_path=args.test_dataset,
        model_name=args.model,
        random_state=args.random_state,
        min_importance_percent=args.min_importance_percent,
        prior_strength=args.prior_strength,
    )

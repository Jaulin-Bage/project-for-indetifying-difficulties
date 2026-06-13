import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

import util
from params import RESULT_VECTOR_MAPPING
from selfOptimization import encode


LABEL_COLUMN = "实际认定结果"


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
    raise ValueError(f"Unsupported model: {model_name}")


def save_feature_importance(model, feature_names, output_dir, model_name):
    if not hasattr(model, "feature_importances_"):
        return None

    importance_df = pd.DataFrame({
        "indicator": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    output_path = output_dir / f"traditional_{model_name}_feature_importance.csv"
    importance_df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return output_path


def train_traditional_model(
    dataset_path,
    indicator_path,
    output_dir,
    test_dataset_path=None,
    model_name="tree",
    test_size=0.2,
    random_state=2024,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    x, y, feature_names = build_feature_frame(dataset_path, indicator_path)
    if test_dataset_path is None:
        stratify = y if y.nunique() > 1 else None
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )
    else:
        x_train, y_train = x, y
        x_test, y_test, _ = build_feature_frame(test_dataset_path, indicator_path)
        x_test = x_test.reindex(columns=feature_names, fill_value=0)

    model = make_model(model_name, random_state)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    test_pred = model.predict(x_test)

    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    report = classification_report(y_test, test_pred, digits=4, zero_division=0)
    matrix = confusion_matrix(y_test, test_pred)

    model_path = output_dir / f"traditional_{model_name}.joblib"
    joblib.dump(model, model_path)
    importance_path = save_feature_importance(model, feature_names, output_dir, model_name)

    report_path = output_dir / f"traditional_{model_name}_report.txt"
    report_path.write_text(
        "\n".join([
            f"Model: {model_name}",
            f"Train samples: {len(x_train)}, test samples: {len(x_test)}",
            f"Train accuracy: {train_acc:.4f}",
            f"Test accuracy: {test_acc:.4f}",
            "",
            "Classification report:",
            report,
        ]),
        encoding="utf-8",
    )

    matrix_path = output_dir / f"traditional_{model_name}_confusion_matrix.csv"
    pd.DataFrame(matrix).to_csv(matrix_path, index=False, encoding="utf-8-sig")

    print(f"Model: {model_name}")
    print(f"Train samples: {len(x_train)}, test samples: {len(x_test)}")
    print(f"Train accuracy: {train_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    print("\nClassification report:")
    print(report)
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
        "model_path": model_path,
        "report_path": report_path,
        "matrix_path": matrix_path,
        "importance_path": importance_path,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Train a traditional ML baseline.")
    parser.add_argument("--dataset", help="CSV dataset with actual labels.")
    parser.add_argument("--train_dataset", help="CSV training dataset with actual labels.")
    parser.add_argument("--test_dataset", help="CSV testing dataset with actual labels.")
    parser.add_argument("--old_zbtx_file", required=True, help="Indicator system CSV.")
    parser.add_argument("--output_dir", default="tmp", help="Directory for model outputs.")
    parser.add_argument(
        "--model",
        choices=["tree", "forest"],
        default="tree",
        help="Traditional model to train.",
    )
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_state", type=int, default=2024)
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
        test_size=args.test_size,
        random_state=args.random_state,
    )

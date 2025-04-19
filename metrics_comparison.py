import json
import math
import os
import re

# Output file paths
BASE_PATH = "model_metrics"
METRICS_ANALYSIS_DIRECTORY = os.path.join(BASE_PATH, "metrics_analysis")
FULL_ANALYSIS_FILE = os.path.join(
    METRICS_ANALYSIS_DIRECTORY, "full_model_comparison.json"
)
POSITIVE_CLASS_ANALYSIS_FILE = os.path.join(
    METRICS_ANALYSIS_DIRECTORY, "positive_class_comparison.json"
)
AVERAGE_METRICS_FILE = os.path.join(
    METRICS_ANALYSIS_DIRECTORY, "model_average_metrics.json"
)

# Dictionary with model file paths and their names
MODEL_FILES = {
    "MLP Without KFold": os.path.join(BASE_PATH, "mlpClassifier/mlp_evaluation.json"),
    "MLP With KFold": os.path.join(
        BASE_PATH, "kFoldMlpClassifier/mlp_kfold_evaluation.json"
    ),
    "xgBoost": os.path.join(BASE_PATH, "xgBoost/xgBoost.json"),
    "LinearSVC": os.path.join(BASE_PATH, "linearSvc/linear_svc.json"),
    "Distilbert": os.path.join(
        BASE_PATH, "distilbertLightingAi/distilbert_metrics.json"
    ),
}

os.makedirs(METRICS_ANALYSIS_DIRECTORY, exist_ok=True)


def load_model_metrics(file_paths):
    """Load metrics from all model files"""
    models_data = {}

    for model_name, file_path in file_paths.items():
        try:
            with open(file_path, "r") as f:
                models_data[model_name] = json.load(f)
                print(f"Successfully loaded data for {model_name}")
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    return models_data


def find_best_model(models_data, metric="f1-score", class_label="1.0", article="1"):
    """Find the best model based on specified metric and class"""
    best_score = -1
    best_model = None

    for model_name, model_data in models_data.items():
        try:
            # Check if the article exists
            if article not in model_data:
                # Default to 0 if article doesn't exist
                continue

            # Check if the class label exists
            if class_label not in model_data[article]:
                # Default to 0 if class label doesn't exist
                continue

            # Check if the metric exists
            if metric not in model_data[article][class_label]:
                # Default to 0 if metric doesn't exist
                continue

            score = model_data[article][class_label][metric]
            if score > best_score:
                best_score = score
                best_model = model_name
        except Exception:
            # Default to 0 for any other exceptions
            pass

    return best_model, best_score


def sort_articles(articles):
    def article_key(article):
        # Match plain number: "1"
        if re.fullmatch(r"\d+", article):
            return (0, int(article), -1, "")

        # Match sub-article: "2-1"
        if re.fullmatch(r"\d+-\d+", article):
            main, sub = map(int, article.split("-"))
            return (0, main, sub, "")

        # Match prefixed articles: "p1", "p1-1", "p2-2"
        match = re.fullmatch(r"([a-zA-Z]+)(\d+)(?:-(\d+))?", article)
        if match:
            prefix, num1, num2 = match.groups()
            num1 = int(num1)
            num2 = int(num2) if num2 else -1
            return (1, num1, num2, prefix)

        # Fallback
        return (2, 0, 0, article)

    return sorted(articles, key=article_key)


def compare_all_metrics(models_data):
    """Create comparison of all models with best and worst metrics per label"""
    comparison = {"models_comparison": {}, "best_and_worst_models": {}}

    all_articles = set()
    for model_data in models_data.values():
        all_articles.update(model_data.keys())

    sorted_articles = sort_articles(all_articles)

    for article in sorted_articles:
        comparison["models_comparison"][article] = {}

        # Collect all models' data for this article
        article_metrics = {}
        for model_name, model_data in models_data.items():
            if article in model_data:
                article_metrics[model_name] = model_data[article]
            else:
                # Add placeholder with default values
                article_metrics[model_name] = {
                    lbl: {
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1-score": 0.0,
                        "support": 0.0,
                    }
                    for lbl in ["0.0", "1.0", "macro avg", "weighted avg"]
                }
                article_metrics[model_name]["accuracy"] = 0.0

        comparison["models_comparison"][article] = article_metrics
        comparison["best_and_worst_models"][article] = {}

        if not article_metrics:
            continue

        # Get valid label keys: must be dicts with precision/recall/f1-score
        example_model = next(iter(article_metrics.values()))
        labels = [
            key
            for key, val in example_model.items()
            if isinstance(val, dict)
            and all(m in val for m in ["precision", "recall", "f1-score"])
        ]

        for label in labels:
            label_metrics_summary = {}

            for metric_name in ["precision", "recall", "f1-score"]:
                best_model = None
                worst_model = None
                best_score = -1
                worst_score = float("inf")
                support = 0.0

                for model_name, model_metric in article_metrics.items():
                    label_data = model_metric.get(label)
                    if not label_data:
                        continue

                    value = label_data.get(metric_name, 0.0)

                    if value > best_score:
                        best_score = value
                        best_model = model_name
                        support = label_data.get("support", 0.0)

                    if value < worst_score:
                        worst_score = value
                        worst_model = model_name

                label_metrics_summary[metric_name] = {
                    "best_model": best_model,
                    "best_score": best_score,
                    "worst_model": worst_model,
                    "worst_score": worst_score,
                    "support": support,
                }

            comparison["best_and_worst_models"][article][label] = label_metrics_summary

    return comparison


def extract_positive_class_metrics(models_data):
    """Extract metrics only for the positive class (label 1.0)"""
    positive_metrics = {"positive_class_comparison": {}, "best_and_worst_models": {}}

    # Get all articles
    all_articles = set()
    for model_data in models_data.values():
        all_articles.update(model_data.keys())

    # Sort articles according to the specified order
    sorted_articles = sort_articles(all_articles)

    # For each article, extract positive class metrics
    for article in sorted_articles:
        positive_metrics["positive_class_comparison"][article] = {}

        for model_name, model_data in models_data.items():
            if article in model_data:
                # If 1.0 exists, add its metrics, otherwise add default metrics with 0 values
                if "1.0" in model_data[article]:
                    positive_metrics["positive_class_comparison"][article][
                        model_name
                    ] = model_data[article]["1.0"]
                else:
                    # Add default metrics with 0 values
                    positive_metrics["positive_class_comparison"][article][
                        model_name
                    ] = {
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1-score": 0.0,
                        "support": 0.0,
                    }

        # Find best model for each metric at this article
        best_precision = {"model": None, "score": -1}
        best_recall = {"model": None, "score": -1}
        best_f1 = {"model": None, "score": -1}

        for model_name, model_data in models_data.items():
            if article in model_data:
                # Default values if 1.0 class or metrics don't exist
                precision = 0.0
                recall = 0.0
                f1 = 0.0

                # Try to get actual values if they exist
                if "1.0" in model_data[article]:
                    precision = model_data[article]["1.0"].get("precision", 0.0)
                    recall = model_data[article]["1.0"].get("recall", 0.0)
                    f1 = model_data[article]["1.0"].get("f1-score", 0.0)

                # Check precision
                if precision > best_precision["score"]:
                    best_precision["score"] = precision
                    best_precision["model"] = model_name

                # Check recall
                if recall > best_recall["score"]:
                    best_recall["score"] = recall
                    best_recall["model"] = model_name

                # Check F1
                if f1 > best_f1["score"]:
                    best_f1["score"] = f1
                    best_f1["model"] = model_name

        # Store best models for this article
        positive_metrics["best_and_worst_models"][article] = {
            "best_precision": best_precision,
            "best_recall": best_recall,
            "best_f1_score": best_f1,
        }

    return positive_metrics


def compute_average_metrics(models_data):
    """Compute average precision, recall, f1-score per label for each model with micro and macro averages"""
    model_averages = {}

    for model_name, model_data in models_data.items():
        # For macro average
        label_sums = {}
        label_counts = {}

        # For micro average
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_samples = 0

        for article, metrics in model_data.items():
            for label, label_metrics in metrics.items():
                if label == "accuracy":
                    continue

                # Skip averages if they exist in the original data
                if label in ["macro avg", "weighted avg"]:
                    continue

                # Get support (number of samples)
                support = label_metrics.get("support", 0)
                if not isinstance(support, (int, float)) or math.isnan(support):
                    support = 0

                # For macro average calculation
                if label not in label_sums:
                    label_sums[label] = {
                        "precision": 0.0,
                        "recall": 0.0,
                        "f1-score": 0.0,
                    }
                    label_counts[label] = 0

                precision_value = label_metrics.get("precision", 0.0)
                precision_value = (
                    precision_value
                    if isinstance(precision_value, (int, float))
                    and not math.isnan(precision_value)
                    else 0.0
                )
                label_sums[label]["precision"] += precision_value

                recall_value = label_metrics.get("recall", 0.0)
                recall_value = (
                    recall_value
                    if isinstance(recall_value, (int, float))
                    and not math.isnan(recall_value)
                    else 0.0
                )
                label_sums[label]["recall"] += recall_value

                f1_value = label_metrics.get("f1-score", 0.0)
                f1_value = (
                    f1_value
                    if isinstance(f1_value, (int, float)) and not math.isnan(f1_value)
                    else 0.0
                )
                label_sums[label]["f1-score"] += f1_value

                label_counts[label] += 1

                # For micro average calculation
                # Calculate true positives, false positives, false negatives from precision and recall
                if precision_value > 0 or recall_value > 0:  # Avoid division by zero
                    if precision_value > 0:
                        # TP / (TP + FP) = precision
                        # TP = precision * (TP + FP)
                        tp_fp = support / recall_value if recall_value > 0 else 0
                        tp = precision_value * tp_fp
                    else:
                        tp = 0

                    if recall_value > 0:
                        # TP / (TP + FN) = recall
                        # TP = recall * (TP + FN)
                        tp_fn = support
                        tp = recall_value * tp_fn
                    else:
                        tp = 0

                    # Use the more reliable TP calculation
                    tp = min(tp, support)

                    # Calculate FP and FN
                    fp = (tp / precision_value) - tp if precision_value > 0 else 0
                    fn = (tp / recall_value) - tp if recall_value > 0 else 0

                    # Accumulate for micro averaging
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                    total_samples += support

        # Calculate macro average metrics
        model_averages[model_name] = {}
        for label in label_sums:
            count = label_counts[label]
            if count > 0:
                model_averages[model_name][label] = {
                    "average_precision": label_sums[label]["precision"] / count,
                    "average_recall": label_sums[label]["recall"] / count,
                    "average_f1_score": label_sums[label]["f1-score"] / count,
                    "article_count": count,
                }
            else:
                model_averages[model_name][label] = {
                    "average_precision": 0.0,
                    "average_recall": 0.0,
                    "average_f1_score": 0.0,
                    "article_count": 0,
                }

        # Calculate micro average metrics
        micro_precision = (
            total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        )
        micro_recall = (
            total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        )
        micro_f1 = (
            2 * (micro_precision * micro_recall) / (micro_precision + micro_recall)
            if (micro_precision + micro_recall) > 0
            else 0.0
        )

        # Add micro averages to the model metrics
        model_averages[model_name]["micro_avg"] = {
            "average_precision": micro_precision,
            "average_recall": micro_recall,
            "average_f1_score": micro_f1,
            "total_samples": total_samples,
        }

        # Calculate macro averages (true definition)
        all_precisions = [
            values["average_precision"]
            for values in model_averages[model_name].values()
            if isinstance(values, dict)
            and "average_precision" in values
            and values != model_averages[model_name].get("micro_avg")
        ]
        all_recalls = [
            values["average_recall"]
            for values in model_averages[model_name].values()
            if isinstance(values, dict)
            and "average_recall" in values
            and values != model_averages[model_name].get("micro_avg")
        ]
        all_f1s = [
            values["average_f1_score"]
            for values in model_averages[model_name].values()
            if isinstance(values, dict)
            and "average_f1_score" in values
            and values != model_averages[model_name].get("micro_avg")
        ]

        # Add macro averages to the model metrics
        model_averages[model_name]["macro_avg"] = {
            "average_precision": (
                sum(all_precisions) / len(all_precisions) if all_precisions else 0.0
            ),
            "average_recall": (
                sum(all_recalls) / len(all_recalls) if all_recalls else 0.0
            ),
            "average_f1_score": sum(all_f1s) / len(all_f1s) if all_f1s else 0.0,
            "class_count": len(all_precisions),
        }

    return model_averages


def main():
    # Load data from all model files
    models_data = load_model_metrics(MODEL_FILES)

    if not models_data:
        print("Error: No model data could be loaded. Please check the file paths.")
        return

    # Generate full comparison
    full_comparison = compare_all_metrics(models_data)

    # Generate positive class comparison
    positive_class_comparison = extract_positive_class_metrics(models_data)

    # Compute average metrics
    model_averages = compute_average_metrics(models_data)

    # Write results to JSON files
    with open(FULL_ANALYSIS_FILE, "w") as f:
        json.dump(full_comparison, f, indent=4)
        print(f"Full analysis saved to {FULL_ANALYSIS_FILE}")

    with open(POSITIVE_CLASS_ANALYSIS_FILE, "w") as f:
        json.dump(positive_class_comparison, f, indent=4)
        print(f"Positive class analysis saved to {POSITIVE_CLASS_ANALYSIS_FILE}")

    with open(AVERAGE_METRICS_FILE, "w") as f:
        json.dump(model_averages, f, indent=4)
        print("Average metrics saved to model_average_metrics.json")


if __name__ == "__main__":
    main()

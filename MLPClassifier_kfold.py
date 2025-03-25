import gc
import json
import os
import pickle
import sqlite3
import time

import numpy
import optuna
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier

# Existing constants
DB_PATH = "echr_cases_anonymized.sqlite"
MODEL_DIR = "echr_mlp_model"
HYPERPARAMETERS = "best_hyperparams.json"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"
SAMPLE_SIZE_FOR_OPTUNA = 1000

# New constants for K-Fold
KFOLD_DIR = os.path.join(MODEL_DIR, "kfold_models")
METRICS_DIR = os.path.join(MODEL_DIR, "metrics")
KFOLD_EVALUATION = "kfold_evaluation.json"
REGULAR_EVALUATION = "regular_evaluation.json"
TOTAL_EVALUATION = "mlp_evaluation.json"
SMALL_SUPPORT_THRESHOLD = (
    200  # Articles with fewer than this many samples will use K-Fold
)
NUM_FOLDS = 5  # Number of folds for cross-validation

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(KFOLD_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# Reuse your existing fetch_cases function
def fetch_cases():
    try:
        print("Connecting to DB...")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        print("Fetching cases...")
        cursor.execute("SELECT case_id, anonymized_judgement FROM cases;")
        result_cases = cursor.fetchall()

        print("Fetching articles...")
        cursor.execute(
            "SELECT normalized_article, case_id FROM articles WHERE normalized_article != '';"
        )
        result_articles = cursor.fetchall()

        cursor.close()
        conn.close()

        df_cases = pd.DataFrame(result_cases, columns=["case_id", "judgment"])
        df_articles = pd.DataFrame(result_articles, columns=["article", "case_id"])

        return df_cases, df_articles

    except sqlite3.Error as error:
        print("Error occurred - ", error)
        exit()


def train_with_kfold(article, article_idx, X, y, best_hyperparams, max_iter=200):
    """
    Train a model for a specific article using K-Fold cross-validation and create an ensemble.

    Args:
        article: The article name/number
        article_idx: The index of the article in the y matrix
        X: The feature matrix
        y: The target matrix
        best_hyperparams: Hyperparameters to use
        max_iter: Maximum iterations for training

    Returns:
        ensemble_models: A list of models to use for ensemble prediction
        evaluation_results: Dict with performance metrics
    """
    print(f"Training model for {article} using {NUM_FOLDS}-Fold cross-validation...")

    # Extract the relevant target vector for this article
    y_article = y[:, article_idx].toarray().ravel().astype(int)

    # Count class occurrences
    class_counts = numpy.bincount(y_article)
    print(f"Class counts for {article}: {class_counts}")

    # Check if we have extremely rare classes
    min_class_count = min(class_counts)
    n_classes = len(class_counts)

    # Variables to keep track of the models and their performance
    ensemble_models = []
    fold_results = []

    # For extremely imbalanced datasets with very few positives
    if min_class_count < 2 or n_classes < 2:
        print(
            f"⚠️ Warning: Article {article} has only {min_class_count} samples for some classes."
        )
        print(f"Using a single DummyClassifier instead of K-Fold CV.")

        # Train a DummyClassifier with a stratified strategy
        model = DummyClassifier(strategy="stratified")
        model.fit(X, y_article)

        # Evaluate on the entire dataset (this is just indicative)
        y_pred = model.predict(X)
        metrics = classification_report(
            y_article, y_pred, digits=4, zero_division=numpy.nan, output_dict=True
        )

        formatted_report = classification_report(
            y_article, y_pred, digits=4, zero_division=numpy.nan
        )
        print(f"Metrics (indicative only):\n{formatted_report}")

        # Store in fold_results format for consistency
        fold_results.append({"fold": 1, "metrics": metrics})

        ensemble_models = [model]  # Just one model in this case

        # Prepare evaluation results
        evaluation_results = {
            "article": article,
            "num_samples": len(y_article),
            "class_distribution": class_counts.tolist(),
            "method": "dummy_classifier",
            "avg_metrics": metrics.get("weighted avg", {}),
            "fold_results": fold_results,
        }

        return ensemble_models, evaluation_results

    # For articles with sufficient data for K-Fold
    # Check if we can use StratifiedKFold or need regular KFold
    if min_class_count >= NUM_FOLDS:
        print(f"Using StratifiedKFold to maintain class distribution...")
        kfold = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)
    else:
        print(
            f"Using regular KFold as class counts ({class_counts}) are too small for stratification..."
        )
        kfold = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

    # Perform K-Fold training and validation
    for fold, (train_idx, val_idx) in enumerate(
        kfold.split(X, y_article if isinstance(kfold, StratifiedKFold) else None)
    ):
        print(f"  Training fold {fold + 1}/{NUM_FOLDS}...")
        start = time.time()

        # Split data for this fold
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y_article[train_idx], y_article[val_idx]

        # Check for class imbalance in this fold
        train_fold_class_counts = numpy.bincount(y_train_fold)
        train_fold_classes = len(train_fold_class_counts)
        min_fold_class_count = min(train_fold_class_counts)

        # Check if this fold has only one class - skip if so
        if train_fold_classes < 2:
            print(f"  ⚠️ Fold {fold + 1}: Contains only one class - skipping this fold")
            fold_results.append(
                {"fold": fold + 1, "metrics": None, "error": "Contains only one class"}
            )
            continue

        # Create model for this fold
        if min_fold_class_count < 2:
            print(
                f"  ⚠️ Fold {fold + 1}: Too few samples of some classes ({train_fold_class_counts})"
            )
            print(f"  Using DummyClassifier for this fold")
            fold_model = DummyClassifier(strategy="stratified")
        else:
            # Use MLPClassifier with modified parameters
            # Disable early_stopping when we have small support to avoid internal validation split
            fold_model = MLPClassifier(
                **best_hyperparams,
                max_iter=max_iter,
                # Only use early_stopping if we have sufficient data
                early_stopping=min_fold_class_count >= 4,
                random_state=42,
                n_iter_no_change=5,
                # Use smaller validation_fraction with small datasets
                validation_fraction=0.1 if min_fold_class_count >= 10 else 0.05,
                verbose=False,
            )

        try:
            # Train the model for this fold
            fold_model.fit(X_train_fold, y_train_fold)

            # Evaluate the model
            y_pred_fold = fold_model.predict(X_val_fold)

            # Get performance metrics
            fold_metrics = classification_report(
                y_val_fold,
                y_pred_fold,
                digits=4,
                zero_division=numpy.nan,
                output_dict=True,
            )

            # Format and print metrics for this fold
            fold_formatted_report = classification_report(
                y_val_fold, y_pred_fold, digits=4, zero_division=numpy.nan
            )

            elapsed_time = time.time() - start
            print(f"  Fold {fold + 1} training time: {elapsed_time:.2f} seconds")
            print(f"  Fold {fold + 1} metrics:\n{fold_formatted_report}")

            # Add this model to the ensemble
            ensemble_models.append(fold_model)

            # Save fold results
            fold_results.append({"fold": fold + 1, "metrics": fold_metrics})

            # Save individual fold model
            fold_model_path = os.path.join(KFOLD_DIR, f"{article}_fold_{fold + 1}.pkl")
            with open(fold_model_path, "wb") as f:
                pickle.dump(fold_model, f)

        except Exception as e:
            print(f"  ⚠️ Error in fold {fold + 1}: {str(e)}")
            print(f"  Skipping this fold")

            # Add empty results for this fold
            fold_results.append({"fold": fold + 1, "metrics": None, "error": str(e)})

        # Clean up fold variables to prevent memory leaks
        # (but keep the model as we need it for the ensemble)
        gc.collect()

    # Check if we have any successful folds
    successful_folds = [f for f in fold_results if f["metrics"] is not None]

    if not successful_folds or not ensemble_models:
        print(f"⚠️ No successful folds for article {article}. Using DummyClassifier.")
        model = DummyClassifier(strategy="stratified")
        model.fit(X, y_article)
        ensemble_models = [model]

        # Evaluate on the entire dataset
        y_pred = model.predict(X)
        metrics = classification_report(
            y_article, y_pred, digits=4, zero_division=numpy.nan, output_dict=True
        )

        evaluation_results = {
            "article": article,
            "num_samples": len(y_article),
            "class_distribution": class_counts.tolist(),
            "method": "dummy_classifier_fallback",
            "avg_metrics": metrics.get("weighted avg", {}),
            "fold_results": fold_results,
            "fallback_metrics": metrics,
        }
    else:
        # Calculate average performance across successful folds
        avg_precision = numpy.mean(
            [
                fold["metrics"].get("weighted avg", {}).get("precision", 0)
                for fold in fold_results
                if fold["metrics"] is not None
            ]
        )
        avg_recall = numpy.mean(
            [
                fold["metrics"].get("weighted avg", {}).get("recall", 0)
                for fold in fold_results
                if fold["metrics"] is not None
            ]
        )
        avg_f1 = numpy.mean(
            [
                fold["metrics"].get("weighted avg", {}).get("f1-score", 0)
                for fold in fold_results
                if fold["metrics"] is not None
            ]
        )

        # Prepare evaluation results
        evaluation_results = {
            "article": article,
            "num_samples": len(y_article),
            "class_distribution": class_counts.tolist(),
            "method": "kfold_ensemble",
            "num_models_in_ensemble": len(ensemble_models),
            "num_successful_folds": len(successful_folds),
            "avg_metrics": {
                "precision": float(avg_precision),
                "recall": float(avg_recall),
                "f1-score": float(avg_f1),
            },
            "fold_results": fold_results,
        }

        print(f"K-Fold CV completed for article {article}")
        print(f"Ensemble created with {len(ensemble_models)} models")
        print(
            f"Average metrics: Precision={avg_precision:.4f}, Recall={avg_recall:.4f}, F1={avg_f1:.4f}"
        )

    return ensemble_models, evaluation_results


# Add this new function to use the ensemble for predictions
def ensemble_predict(models, X):
    """
    Make predictions using an ensemble of models.

    Args:
        models: List of trained models
        X: Feature matrix to predict on

    Returns:
        predictions: The ensemble predictions
    """
    if not models:
        raise ValueError("No models provided for ensemble prediction")

    # Check if models can predict probabilities
    can_predict_proba = all(hasattr(model, "predict_proba") for model in models)

    if can_predict_proba:
        # Get probability predictions from all models
        all_probs = []
        for model in models:
            prob = model.predict_proba(X)
            all_probs.append(prob)

        # Average the probabilities
        avg_probs = numpy.mean(all_probs, axis=0)

        # Convert to class predictions (argmax)
        predictions = numpy.argmax(avg_probs, axis=1)
    else:
        # Get class predictions from all models
        all_preds = []
        for model in models:
            pred = model.predict(X)
            all_preds.append(pred)

        # Stack predictions and take majority vote
        all_preds = numpy.vstack(all_preds)
        # Take mode (majority vote) along axis 0 (across models)
        predictions = numpy.apply_along_axis(
            lambda x: numpy.bincount(x).argmax(), axis=0, arr=all_preds
        )

    return predictions


def identify_small_support_articles(y_train, articles):
    """
    Identify articles with small support (few samples) that would benefit from K-Fold CV.

    Returns:
        list of (article, index) tuples for articles with small support
    """
    small_support_articles = []
    regular_support_articles = []

    for idx, article in enumerate(articles):
        y_article = y_train[:, idx].toarray().ravel()
        positive_samples = numpy.sum(y_article)

        if positive_samples < SMALL_SUPPORT_THRESHOLD:
            small_support_articles.append((article, idx))
            print(
                f"Article {article} has {positive_samples} positive samples (less than threshold {SMALL_SUPPORT_THRESHOLD})"
            )
        else:
            regular_support_articles.append((article, idx))
            print(
                f"Article {article} has {positive_samples} positive samples (more than threshold {SMALL_SUPPORT_THRESHOLD})"
            )

    print(
        f"Found {len(small_support_articles)} articles with small support and {len(regular_support_articles)} with regular support"
    )
    return small_support_articles, regular_support_articles


def create_optimization_subset(X_train, y_train, sample_size=1000):
    """Create a smaller subset for hyperparameter optimization."""
    if X_train.shape[0] > sample_size:
        indices = numpy.random.choice(X_train.shape[0], sample_size, replace=False)
        return X_train[indices], y_train[indices]
    return X_train, y_train


def evaluate_model(model, article, X_val, y_val, idx):
    y_pred = model.predict(X_val)

    score_dict = classification_report(
        y_val[:, idx].toarray().ravel(),
        y_pred,
        digits=4,
        zero_division=numpy.nan,
        output_dict=True,
    )

    # Print formatted classification report
    score = classification_report(
        y_val[:, idx].toarray().ravel(), y_pred, digits=4, zero_division=numpy.nan
    )
    print(f"Article {article}:\n {score}")

    return score_dict


def objective(trial, X_train, y_train):
    """Define Optuna's optimization function for MLPClassifier with pruning."""
    # Instead of using tuples directly, use string encoding and convert later
    hidden_layer_option = trial.suggest_categorical(
        "hidden_layer_option", ["100", "100_50", "100_100", "200", "200_100"]
    )

    # Convert string representation to actual tuple when creating the model
    if "_" in hidden_layer_option:
        hidden_layer_sizes = tuple(int(x) for x in hidden_layer_option.split("_"))
    else:
        hidden_layer_sizes = (int(hidden_layer_option),)

    # MLP parameters to optimize
    activation = trial.suggest_categorical("activation", ["relu", "tanh"])
    solver = trial.suggest_categorical("solver", ["adam", "sgd"])
    alpha = trial.suggest_float("alpha", 1e-5, 1e-2, log=True)
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
    max_iter = trial.suggest_int("max_iter", 100, 300)

    # For optimization, we'll use just the first article
    article_index = 0
    y_target = y_train[:, article_index].toarray().ravel()

    # Create and train the MLP model
    mlp = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate_init=learning_rate_init,
        max_iter=max_iter,
        early_stopping=True,
        n_iter_no_change=10,
        validation_fraction=0.1,
        random_state=42,
    )

    try:
        mlp.fit(X_train, y_target)
        # For early stopping, use the best validation score if available
        if hasattr(mlp, "best_validation_score_"):
            score = (
                mlp.best_validation_score_ * -1
            )  # Convert to negative for minimization
        else:
            # If no early stopping occurred, use the final loss
            score = mlp.loss_
    except Exception as e:
        print(f"Error during MLP training: {e}")
        return float("inf")  # Return high value for failed trials

    # Explicitly force garbage collection
    gc.collect()

    return score


def main():
    """Main function to run the K-Fold implementation for small support articles"""
    # Load and preprocess data
    df_cases, df_articles = fetch_cases()

    # Merge cases with articles (One-hot encode article labels)
    merged_df = df_cases.merge(df_articles, on="case_id", how="left")
    pivot_df = pd.crosstab(
        df_articles["case_id"], df_articles["article"], values=1, aggfunc="count"
    ).fillna(0)
    pivot_df = (pivot_df > 0).astype(int)
    cases_to_articles_df = df_cases.merge(pivot_df, on="case_id", how="left").fillna(0)

    ARTICLES_COLUMNS = pivot_df.columns.tolist()
    print(f"Total articles found: {len(ARTICLES_COLUMNS)}")

    # Remove cases with no linked articles
    cases_to_articles_df["num_articles"] = cases_to_articles_df[ARTICLES_COLUMNS].sum(
        axis=1
    )
    cases_to_articles_df = cases_to_articles_df[
        cases_to_articles_df["num_articles"] > 0
    ]
    cases_to_articles_df.drop("num_articles", axis=1, inplace=True)

    print("Final dataset shape:", cases_to_articles_df.shape)

    # Split Data into Train and Validation
    train_df, val_df = train_test_split(
        cases_to_articles_df, test_size=0.2, random_state=42
    )
    print("Train shape:", train_df.shape)
    print("Validation shape:", val_df.shape)

    # Convert Text to Features (TF-IDF)
    vectorizer = TfidfVectorizer(max_features=100000, stop_words="english")
    X_train = csr_matrix(vectorizer.fit_transform(train_df["judgment"]))
    X_val = csr_matrix(vectorizer.transform(val_df["judgment"]))
    y_train = csr_matrix(train_df[ARTICLES_COLUMNS].values)
    y_val = csr_matrix(val_df[ARTICLES_COLUMNS].values)
    print("TF-IDF feature shape:", X_train.shape)

    print("Saving vectorizer...")
    vectorizer_path = os.path.join(MODEL_DIR, VECTORIZER_FILE)
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)

    # Run Optuna Tuning
    if not os.path.exists(os.path.join(MODEL_DIR, HYPERPARAMETERS)):
        print("\nStarting Hyperparameter Tuning with Optuna and Pruning...")
        X_train_opt, y_train_opt = create_optimization_subset(
            X_train, y_train, SAMPLE_SIZE_FOR_OPTUNA
        )
        study = optuna.create_study(
            direction="minimize",  # Minimize negative validation score or loss
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
        )
        study.optimize(
            lambda trial: objective(trial, X_train_opt, y_train_opt),
            n_trials=20,
            show_progress_bar=True,
            gc_after_trial=True,
        )

        # Save best hyperparameters
        best_params = study.best_params

        # Convert hidden_layer_option back to hidden_layer_sizes
        hidden_layer_option = best_params.pop("hidden_layer_option")
        if "_" in hidden_layer_option:
            hidden_layer_sizes = tuple(int(x) for x in hidden_layer_option.split("_"))
        else:
            hidden_layer_sizes = (int(hidden_layer_option),)

        # Create the hyperparameters dict with the correctly formatted hidden_layer_sizes
        best_hyperparams = {"hidden_layer_sizes": hidden_layer_sizes, **best_params}

        with open(os.path.join(MODEL_DIR, HYPERPARAMETERS), "w") as f:
            json.dump(
                best_hyperparams,
                f,
                default=lambda obj: (
                    obj
                    if isinstance(obj, (int, float, str, bool, type(None)))
                    else str(obj)
                ),
            )
        print("Best hyperparameters saved:", best_hyperparams)

        del study
        gc.collect()

    # Load best hyperparameters
    with open(os.path.join(MODEL_DIR, HYPERPARAMETERS)) as f:
        print("Loading hyperparameters...")
        best_hyperparams = json.load(f)

    # Get max_iter from hyperparameters (otherwise set a default)
    max_iter = best_hyperparams.pop("max_iter", 200)

    # Convert hidden_layer_sizes from string back to tuple if needed
    if "hidden_layer_sizes" in best_hyperparams:
        if isinstance(best_hyperparams["hidden_layer_sizes"], str):
            # Parse string representation like "(100, 50)"
            best_hyperparams["hidden_layer_sizes"] = eval(
                best_hyperparams["hidden_layer_sizes"]
            )
        elif isinstance(best_hyperparams["hidden_layer_sizes"], list):
            # Convert list to tuple
            best_hyperparams["hidden_layer_sizes"] = tuple(
                best_hyperparams["hidden_layer_sizes"]
            )

    # Identify articles with small support
    small_support_articles, regular_support_articles = identify_small_support_articles(
        y_train, ARTICLES_COLUMNS
    )

    # Dictionary to store evaluation results for all articles
    kfold_evaluation_results = {}
    regular_evaluation_results = {}
    evaluation_results = {}

    # Train models for small support articles using K-Fold CV
    for article, idx in small_support_articles:
        # Check if we already have an ensemble model for this article
        ensemble_path = os.path.join(KFOLD_DIR, f"ensemble_{article}.pkl")
        if os.path.exists(ensemble_path):
            print(f"Ensemble model for {article} already exists. Skipping...")
            continue

        try:
            # Train with k-fold and get ensemble models
            ensemble_models, eval_results = train_with_kfold(
                article, idx, X_train, y_train, best_hyperparams, max_iter
            )

            # Save the ensemble models
            with open(ensemble_path, "wb") as f:
                pickle.dump(ensemble_models, f)

            # Store evaluation results
            kfold_evaluation_results[article] = eval_results

        except Exception as e:
            print(f"❌ Error processing article {article}: {str(e)}")
            # Log the error in the evaluation results
            kfold_evaluation_results[article] = {
                "article": article,
                "error": str(e),
                "status": "failed",
            }
        finally:
            # Clean up
            if "ensemble_models" in locals():
                del ensemble_models
            gc.collect()

    # Evaluate the ensemble models against the validation set
    print("\nEvaluating ensemble models on validation set...")

    kfold_validation_results = {}
    for article, idx in small_support_articles:
        ensemble_path = os.path.join(KFOLD_DIR, f"ensemble_{article}.pkl")
        if os.path.exists(ensemble_path):
            try:
                # Load the ensemble models
                with open(ensemble_path, "rb") as f:
                    ensemble_models = pickle.load(f)

                # Get the target column for this article
                y_val_article = y_val[:, idx].toarray().ravel()

                # Make predictions using the ensemble
                y_pred = ensemble_predict(ensemble_models, X_val)

                # Calculate metrics
                validation_metrics = classification_report(
                    y_val_article,
                    y_pred,
                    digits=4,
                    zero_division=numpy.nan,
                    output_dict=True,
                )

                # Print formatted report
                print(
                    f"\nValidation metrics for article {article} (ensemble of {len(ensemble_models)} models):"
                )
                print(
                    classification_report(
                        y_val_article, y_pred, digits=4, zero_division=numpy.nan
                    )
                )

                # Store results
                kfold_validation_results[article] = validation_metrics

                # Free memory
                del ensemble_models
                gc.collect()

            except Exception as e:
                print(f"❌ Error evaluating ensemble for article {article}: {str(e)}")
                kfold_validation_results[article] = {
                    "error": str(e),
                    "status": "evaluation_failed",
                }

    # Save validation results
    with open(os.path.join(METRICS_DIR, "ensemble_validation_results.json"), "w") as f:
        json.dump(kfold_validation_results, f, indent=4)

    print(
        f"Ensemble models metrics saved to {os.path.join(METRICS_DIR, 'ensemble_validation_results.json')}"
    )

    for article, idx in regular_support_articles:
        # Check if the model already exists
        model_path = os.path.join(MODEL_DIR, f"mlp_{article}.pkl")
        if os.path.exists(model_path):
            print(
                f"Model for {article} ({idx + 1}/{len(ARTICLES_COLUMNS)}) already exists. Skipping..."
            )
            # Load the existing model to use for evaluation
            with open(model_path, "rb") as f:
                model = pickle.load(f)

            # Perform evaluation for the existing model
            print("\nEvaluating existing model...")
            regular_evaluation_results[article] = evaluate_model(
                model, article, X_val, y_val, idx
            )
            del model
            gc.collect()
            continue

        print(f"Training model for {article} ({idx + 1}/{len(ARTICLES_COLUMNS)})...")
        start = time.time()

        y_train_article = (
            y_train[:, idx].toarray().ravel().astype(int)
        )  # Convert to int

        # Debug print before bincount
        print(
            f"Unique classes in y_train_article for {article}: {numpy.unique(y_train_article)}"
        )

        # Count class occurrences
        class_counts = numpy.bincount(y_train_article)

        # Debug print after bincount
        print(f"Class counts for {article}: {class_counts}")

        # Check if stratification is possible
        min_class_count = class_counts.min()

        # Define threshold for using DummyClassifier
        dummy_threshold = (
            2  # If any class has fewer than 2 samples, use DummyClassifier
        )

        if min_class_count < dummy_threshold:
            print(
                f"⚠️ Warning: Article {article} has an extremely imbalanced dataset (class counts: {class_counts})."
            )
            print(
                f"Using DummyClassifier to handle this case instead of MLPClassifier."
            )

            # Train a DummyClassifier with a most_frequent strategy
            model = DummyClassifier(strategy="stratified")

        else:
            # Use MLPClassifier as usual
            model = MLPClassifier(
                **best_hyperparams,
                max_iter=max_iter,
                early_stopping=True,
                random_state=42,
                n_iter_no_change=5,
                verbose=True,
            )

        model.fit(X_train, y_train_article)
        elapsed_time = time.time() - start
        minutes = int(elapsed_time // 60)
        seconds = elapsed_time % 60
        print(f"Training Time: {minutes} minutes and {seconds:.2f} seconds")

        print("\nEvaluating model...")
        regular_evaluation_results[article] = evaluate_model(
            model, article, X_val, y_val, idx
        )

        print("Saving model...")
        model_path = os.path.join(MODEL_DIR, f"mlp_{article}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        # Free memory after saving model
        del model
        gc.collect()

    print("Saving regular model evaluations...")
    with open(os.path.join(METRICS_DIR, "regular_evaluation.json"), "w") as f:
        json.dump(regular_evaluation_results, f, indent=4)

    print(
        f"Regular models metrics saved to {os.path.join(METRICS_DIR, 'regular_evaluation.json')}"
    )

    print("Merging model evaluations...")
    evaluation_results = regular_evaluation_results | kfold_validation_results
    evaluation_results = {
        key: val
        for key, val in sorted(evaluation_results.items(), key=lambda ele: ele[0])
    }

    print("Saving merged model evaluations...")
    with open(os.path.join(METRICS_DIR, "mlp_evaluation.json"), "w") as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"Metrics saved to {os.path.join(METRICS_DIR, 'mlp_evaluation.json')}")


if __name__ == "__main__":
    main()

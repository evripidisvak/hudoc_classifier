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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

DB_PATH = "echr_cases_anonymized.sqlite"
MODEL_DIR = "echr_mlp_model"
HYPERPARAMETERS = "best_hyperparams.json"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"
SAMPLE_SIZE_FOR_OPTUNA = 1000
os.makedirs(MODEL_DIR, exist_ok=True)


# Load Data
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

def objective(trial, X_train, y_train):
    """Define Optuna's optimization function for MLPClassifier with pruning."""
    # MLP parameters to optimize
    params = {
            "hidden_layer_sizes": trial.suggest_categorical(
                    "hidden_layer_sizes",
                    [(100,), (100, 50), (100, 100), (200,), (200, 100)]
                    ),
            "activation": trial.suggest_categorical(
                    "activation", ["relu", "tanh"]
                    ),
            "solver": trial.suggest_categorical(
                    "solver", ["adam", "sgd"]
                    ),
            "alpha": trial.suggest_float(
                    "alpha", 1e-5, 1e-2, log=True
                    ),
            "learning_rate_init": trial.suggest_float(
                    "learning_rate_init", 1e-4, 1e-2, log=True
                    ),
            "max_iter": trial.suggest_int(
                    "max_iter", 100, 300
                    ),
            "early_stopping": True,
            "n_iter_no_change": 10,
            "validation_fraction": 0.1,
            "random_state": 42,
            }

    # For optimization, we'll use just the first article
    article_index = 0
    y_target = y_train[:, article_index].toarray().ravel()

    # Create and train the MLP model
    mlp = MLPClassifier(**params)

    try:
        mlp.fit(X_train, y_target)
        score = mlp.best_validation_score_ * -1  # Convert to negative for minimization
    except Exception as e:
        print(f"Error during MLP training: {e}")
        return float('inf')  # Return high value for failed trials

    # Explicitly force garbage collection
    gc.collect()

    return score


def create_optimization_subset(X_train, y_train, sample_size=1000):
    """Create a smaller subset for hyperparameter optimization."""
    if X_train.shape[0] > sample_size:
        indices = numpy.random.choice(X_train.shape[0], sample_size, replace=False)
        return X_train[indices], y_train[indices]
    return X_train, y_train

def evaluate_models(models, X_val, y_val, articles):
    scores = {}

    for idx, article in enumerate(articles):
        model = models[article]
        y_pred = model.predict(X_val)

        score_dict = classification_report(
                y_val[:, idx].toarray().ravel(),
                y_pred,
                digits=4,
                zero_division=numpy.nan,
                output_dict=True,
                )
        print(f"Article {article}:\n {json.dumps(score_dict, indent=4)}")
        scores[article] = score_dict

    return scores

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


if __name__ == "__main__":
    # Preprocessing Data
    df_cases, df_articles = fetch_cases()

    # Merge cases with articles (One-hot encode article labels)
    merged_df = df_cases.merge(df_articles, on="case_id", how="left")
    pivot_df = pd.crosstab(
            df_articles["case_id"],
            df_articles["article"],
            values=1,  # Set value to 1 for all matches
            aggfunc='count'  # Count occurrences
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
                direction="minimize",  # Minimize negative validation score
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
                )
        study.optimize(
                lambda trial: objective(trial, X_train_opt, y_train_opt),
                n_trials=20,
                show_progress_bar=True,
                gc_after_trial=True,
                )

        # Save best hyperparameters
        best_hyperparams = study.best_params
        with open(os.path.join(MODEL_DIR, HYPERPARAMETERS), "w") as f:
            json.dump(best_hyperparams, f)
        print("Best hyperparameters saved:", best_hyperparams)

    with open(os.path.join(MODEL_DIR, HYPERPARAMETERS)) as f:
        print("Loading hyperparameters...")
        best_hyperparams = json.load(f)

    # Train MLPClassifier Models with Best Parameters
    models = {}
    scores = {}
    evaluation_results = {}

    # Get max_iter from hyperparameters (otherwise set a default)
    max_iter = best_hyperparams.pop("max_iter", 200)

    # Convert hidden_layer_sizes back to tuple if it was serialized as a list
    if "hidden_layer_sizes" in best_hyperparams and isinstance(best_hyperparams["hidden_layer_sizes"], list):
        best_hyperparams["hidden_layer_sizes"] = tuple(best_hyperparams["hidden_layer_sizes"])

    for idx, article in enumerate(ARTICLES_COLUMNS):
        print(f"Training model for {article} ({idx + 1}/{len(ARTICLES_COLUMNS)})...")
        start = time.time()

        y_train_article = y_train[:, idx].toarray().ravel()

        # Create and train the model
        model = MLPClassifier(
                **best_hyperparams,
                max_iter=max_iter,
                early_stopping=True,
                random_state=42,
                verbose=False
                )

        model.fit(X_train, y_train_article)
        models[article] = model
        print("Training Time: %s seconds" % (str(time.time() - start)))

        print("\nEvaluating model...")
        evaluation_results[article] = evaluate_model(
                model, article, X_val, y_val, idx
                )

        print("Saving model...")
        model_path = os.path.join(MODEL_DIR, f"mlp_{article}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    with open("mlp_evaluation.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"Models saved in {MODEL_DIR}")
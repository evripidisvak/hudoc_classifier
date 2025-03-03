import gc
import json
import os
import pickle
import sqlite3
import time

import numpy
import optuna
import pandas as pd
import xgboost as xgb
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

DB_PATH = "echr_cases_anonymized.sqlite"
MODEL_DIR = "echr_xgboost_model"
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
    """Define Optuna's optimization function for XGBoost with pruning."""
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",  # For faster training with sparse data
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        # Memory optimization parameters
        "max_bin": trial.suggest_int("max_bin", 128, 256),  # Reduced number of bins
        "grow_policy": "lossguide",  # More memory-efficient than "depthwise"
        "max_leaves": trial.suggest_int("max_leaves", 32, 64),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 7),
    }

    # Get number of boosting rounds (trees) from trial
    num_boost_round = trial.suggest_int("num_boost_round", 100, 500)

    # Select first article for optimization
    article_index = 0
    y_target = y_train[:, article_index].toarray().ravel()

    # Convert to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_target)

    # Use cross-validation for more robust evaluation
    cv_results = xgb.cv(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        nfold=3,
        early_stopping_rounds=10,
        verbose_eval=False,
    )

    # Explicitly delete DMatrix and force garbage collection
    del dtrain
    gc.collect()

    best_score = cv_results["test-logloss-mean"].min()

    # Report intermediate values for pruning
    for epoch, (train_score, valid_score) in enumerate(
        zip(cv_results["train-logloss-mean"], cv_results["test-logloss-mean"])
    ):
        trial.report(valid_score, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return best_score


def create_optimization_subset(X_train, y_train, sample_size=1000):
    """Create a smaller subset for hyperparameter optimization."""
    if X_train.shape[0] > sample_size:
        indices = numpy.random.choice(X_train.shape[0], sample_size, replace=False)
        return X_train[indices], y_train[indices]
    return X_train, y_train

def evaluate_models(models, X_val, y_val):
    scores = {}
    dval = xgb.DMatrix(X_val)  # Convert once, use for all models

    for idx, article in enumerate(ARTICLES_COLUMNS):
        model = models[article]
        y_pred = model.predict(dval)
        y_pred = (y_pred > 0.5).astype(int)  # Convert to binary labels
        # y_prob = model.predict_proba(dval)[:, 1]

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

def evaluate_model(model, article, X_val, y_val, dval_dmatrix):
    y_pred = model.predict(dval_dmatrix)
    y_pred = (y_pred > 0.5).astype(int)  # Convert to binary labels

    score_dict = classification_report(
        y_val[:, idx].toarray().ravel(),
        y_pred,
        digits=4,
        zero_division=numpy.nan,
        output_dict=True,
    )
    # print(f"Article {article}:\n {json.dumps(score_dict, indent=4)}")
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
            direction="minimize",  # Minimize log loss
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

    # Train XGBoost Models with Best Parameters
    models = {}
    dval_dmatrix = xgb.DMatrix(X_val)
    scores = {}
    evaluation_results = {}
    num_boost_round = best_hyperparams.pop("num_boost_round", 100)

    for idx, article in enumerate(ARTICLES_COLUMNS):
        print(f"Training model for {article} ({idx + 1}/{len(ARTICLES_COLUMNS)})...")
        start = time.time()

        y_train_article = y_train[:, idx].toarray().ravel()
        y_val_article = y_val[:, idx].toarray().ravel()

        dtrain = xgb.DMatrix(X_train, label=y_train_article)
        dval = xgb.DMatrix(X_val, label=y_val_article)

        model = xgb.train(
            best_hyperparams,
            dtrain,
            num_boost_round=num_boost_round,
            evals=[(dval, "validation")],
            verbose_eval=0,
        )
        models[article] = model
        print("Training Time: %s seconds" % (str(time.time() - start)))

        print("\nEvaluating model...")
        evaluation_results[article] = evaluate_model(
            model, article, X_val, y_val, dval_dmatrix
        )

        print("Saving model...")
        model_path = os.path.join(MODEL_DIR, f"xgboost_{article}.ubj")
        model.save_model(model_path)

    with open("xgBoost.json", "w") as f:
        json.dump(evaluation_results, f, indent=4)

    print(f"Models saved in {MODEL_DIR}")

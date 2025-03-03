import argparse
import os
import pickle
import sqlite3
import sys

import numpy

# Fixed paths for models directory and database connection
MODEL_DIR = "echr_linear_svc_model"
DB_PATH = "echr_cases_anonymized.sqlite"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"


class DummyClassifier:
    def __init__(self, constant_class):
        self.constant_class = constant_class

    def predict(self, X):
        return numpy.full(X.shape[0], self.constant_class)

    def predict_proba(self, X):
        probs = numpy.zeros((X.shape[0], 2))
        probs[:, self.constant_class] = 1.0
        return probs


def load_model(model_path):
    """Load a trained LinearSVC model from disk."""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        # print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model from {model_path}: {e}")
        return None


def get_judgment(judgment_id=None):
    """Fetch a specific judgment or a random one from the database."""
    try:
        print("Connecting to DB...")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        if judgment_id:
            cursor.execute(f"SELECT * FROM cases WHERE case_id = '{judgment_id}'")
            result = cursor.fetchone()

            if not result:
                print(f"No judgment found with ID {judgment_id}")
                return None
        else:
            # Get a random judgment
            cursor.execute("SELECT * FROM cases ORDER BY RANDOM() LIMIT 1")
            result = cursor.fetchone()

            if not result:
                print("No judgments found in the database")
                return None

        # Convert row to dict
        judgment = {'case_id': result[1], 'judgement': result[2]}
        # Get associated articles
        cursor.execute(
                f"SELECT normalized_article FROM articles WHERE normalized_article != '' AND case_id = '{judgment['case_id']}';")
        result_articles = cursor.fetchall()
        judgment["articles"] = [item for t in result_articles for item in t]

        return judgment
    except Exception as e:
        print(f"Error fetching judgment: {e}")
        return None


def preprocess_judgment(judgment):
    try:
        vectorizer_path = os.path.join(MODEL_DIR, VECTORIZER_FILE)
        # Load the same TF-IDF vectorizer used during training
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)

        # Extract the judgment text (using the same field name as in training)
        judgment_text = judgment.get('judgement', '')
        if not judgment_text:
            raise Exception('No judgment text')

        # Transform the text using the same vectorizer as during training
        vectorized = vectorizer.transform([judgment_text])

        print(f"Successfully preprocessed judgment (features: {vectorized.shape[1]})")
        return vectorized

    except Exception as e:
        print(f"Error preprocessing judgment: {e}")
        return None


def predict_with_models(models, vectorized_input):
    print("Predicting related articles...")
    predictions = {}
    for model_name, model in models.items():
        try:
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(vectorized_input)[0, 1]
            else:
                # Use decision_function if predict_proba is unavailable
                decision = model.decision_function(vectorized_input)
                proba = 1 / (1 + numpy.exp(-decision))  # Sigmoid function to estimate probability

            predictions[model_name] = proba.item()
        except Exception as e:
            print(f"Error predicting with model {model_name}: {e}")
            predictions[model_name] = None
    return predictions


def display_results(judgment, predictions):
    """Display the judgment information, predictions, and related articles."""
    print("\n" + "=" * 50)
    print(f"Judgment Information")
    print("=" * 50)
    print(f"Judgment id: {judgment['case_id']}")
    print(f"Related articles: {judgment['articles']}")

    print("=" * 50)
    print(f"\nPredictions")
    print("=" * 50)

    print("Predicted articles (probability >= 0.5)")
    for article, probability in predictions.items():
        if probability is not None and probability >= 0.5:
            print(f"Article {article}: {round(probability, 4)}")

    print("=" * 50)

    print("Related articles to predicted probability")
    for rel_article in judgment['articles']:
        if rel_article in predictions and predictions[rel_article] is not None:
            print(f"Article {rel_article}: {round(predictions[rel_article], 4)}")
        else:
            print(f"Article {rel_article}: No prediction available")

    print("=" * 50)
    print("\n" + "=" * 50 + "\n")


sys.modules["__main__"].DummyClassifier = DummyClassifier

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test LinearSVC models against judgments')
    parser.add_argument('--judgment_id', type=str, help='Specific judgment ID to test (random if not provided)')
    args = parser.parse_args()

    # Load all models from the directory
    models = {}
    for file in os.listdir(MODEL_DIR):
        if file.startswith('linear_svc_') and file.endswith('.pkl'):
            model_path = os.path.join(MODEL_DIR, file)
            model_name = os.path.splitext(file)[0].split('_')[2]  # Extract article name
            model = load_model(model_path)
            if model:
                models[model_name] = model

    if not models:
        print("No models loaded, exiting")
        exit()

    print(f"Loaded {len(models)} models: {', '.join(models.keys())}")

    # Get judgment
    judgment = get_judgment(args.judgment_id)
    if not judgment:
        print("No judgment found. Exiting...")
        exit()

    # Preprocess judgment data
    vectorized_input = preprocess_judgment(judgment)
    if vectorized_input is None:
        print("Failed to preprocess judgment. Exiting...")
        exit()

    # Make predictions
    predictions = predict_with_models(models, vectorized_input)

    # Display results
    display_results(judgment, predictions)

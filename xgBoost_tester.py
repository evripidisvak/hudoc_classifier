import pickle
import sqlite3

import xgboost as xgb
import os
import argparse

# Fixed paths for models directory and database connection
MODEL_DIR = "echr_xgboost_model"
DB_PATH = "echr_cases_anonymized.sqlite"
VECTORIZER_FILE = "tfidf_vectorizer.pkl"

def load_model(model_path):
    """Load a trained XGBoost model from disk."""
    try:
        model = xgb.Booster()
        model.load_model(model_path)
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
        vectorizer_path=os.path.join(MODEL_DIR, VECTORIZER_FILE)
        # Load the same TF-IDF vectorizer used during training
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)

        # Extract the judgment text (using the same field name as in training)
        judgment_text = judgment.get('judgement', '')
        if not judgment_text:
            raise Exception('No judgment text')

        # Transform the text using the same vectorizer as during training
        vectorized = vectorizer.transform([judgment_text])

        # Convert to DMatrix format required by XGBoost
        dmatrix = xgb.DMatrix(vectorized)

        print(f"Successfully preprocessed judgment (features: {vectorized.shape[1]})")
        return dmatrix

    except Exception as e:
        print(f"Error preprocessing judgment: {e}")
        return

def map_court_level(court_name):
    """Example helper function to map court names to numeric levels."""
    court_levels = {
            "Supreme Court": 3,
            "High Court": 2,
            "District Court": 1
            }
    return court_levels.get(court_name, 0)  # Default to 0 if unknown

def predict_with_models(models, dmatrix):
    print("Predicting related articles...")
    predictions = {}
    for model_name, model in models.items():
        try:
            pred = model.predict(dmatrix)
            if len(pred) == 1:
                predictions[model_name] = float(pred[0])
            else:
                predictions[model_name] = pred.tolist()
        except Exception as e:
            print(f"Error predicting with model {model_name}: {e}")
            predictions[model_name] = None
    return predictions

def display_results(judgment, predictions):
    """Display the judgment information, predictions, and related articles."""
    print("\n" + "="*50)
    print(f"Judgment Information")
    print("="*50)
    print(f"Judgment id: {judgment['case_id']}")
    print(f"Related articles: {judgment['articles']}")

    print("="*50)
    print(f"\nPredictions")
    print("="*50)

    print("Predicted articles (probability >= 0.5)")
    for article, probability in predictions.items():
        if probability >= 0.5:
            print(f"Article {article}: {round(probability, 4)}")

    print("="*50)

    print("Related articles to predicted probability")
    for rel_article in judgment['articles']:
        print(f"Article {rel_article}: {round(predictions[rel_article], 4)}")

    print("="*50)
    # print(f"Complete predictions:\n {predictions}")
    # print(f"Complete predictions:\n")
    # for article, probability in predictions.items():
    #     print(f"Article {article}: {round(probability, 4)}")
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test XGBoost models against judgments')
    parser.add_argument('--judgment_id', type=str, help='Specific judgment ID to test (random if not provided)')
    args = parser.parse_args()

    # Load all models from the directory
    models = {}
    for file in os.listdir(MODEL_DIR):
        if file.startswith('xgboost_') and file.endswith('.ubj'):
            model_path = os.path.join(MODEL_DIR, file)
            model_name = os.path.splitext(file)[0].split('_')[1]
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
        print("No judgment found. Existing...")
        exit()

    # Preprocess judgment data
    dmatrix = preprocess_judgment(judgment)
    if not dmatrix:
        print("No dmatrix created. Existing...")
        exit()

    # Make predictions
    predictions = predict_with_models(models, dmatrix)

    # Display results
    display_results(judgment, predictions)
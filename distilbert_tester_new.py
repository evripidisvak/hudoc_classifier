import os
import json
import torch
import sqlite3
import numpy as np
import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Import configurations from the training script
from distilbert import (
    DB_PATH,
    MODEL_DIR,
    DATASETS_DIR,
    device
    )

def load_trained_model_and_resources():
    """
    Load the trained model, tokenizer, and article list

    Returns:
    --------
    tuple: (model, tokenizer, articles)
    """
    # Load articles list
    with open(os.path.join(DATASETS_DIR, "articles.json"), "r") as f:
        articles = json.load(f)

    # Load tokenizer and model
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    model = DistilBertForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()  # Set to evaluation mode

    return model, tokenizer, articles

def get_random_case(include_true_articles=True):
    """
    Retrieve a random case from the database

    Parameters:
    -----------
    include_true_articles: bool
        If True, also retrieve the true articles for the case

    Returns:
    --------
    dict: Case information including text and optionally true articles
    """
    conn = sqlite3.connect(DB_PATH)

    if include_true_articles:
        # Join cases with articles to get true article labels
        query = """
        SELECT c.case_id, c.anonymized_judgement, 
               GROUP_CONCAT(DISTINCT a.normalized_article) as articles
        FROM cases c
        JOIN articles a ON c.case_id = a.case_id
        GROUP BY c.case_id
        ORDER BY RANDOM()
        LIMIT 1
        """
        df = pd.read_sql(query, conn)
        case = {
                'case_id': df.iloc[0]['case_id'],
                'text': df.iloc[0]['anonymized_judgement'],
                'true_articles': df.iloc[0]['articles'].split(',')
                }
    else:
        # Just get a random case text
        query = "SELECT case_id, anonymized_judgement FROM cases ORDER BY RANDOM() LIMIT 1"
        df = pd.read_sql(query, conn)
        case = {
                'case_id': df.iloc[0]['case_id'],
                'text': df.iloc[0]['anonymized_judgement']
                }

    conn.close()
    return case

def predict_case_articles(model, tokenizer, case_text, articles, threshold=0.5):
    """
    Predict articles for a given case text

    Parameters:
    -----------
    model: Trained DistilBERT model
    tokenizer: Tokenizer
    case_text: str, input case text to classify
    articles: list, list of possible articles
    threshold: float, probability threshold for positive prediction

    Returns:
    --------
    dict: Detailed prediction results
    """
    # Tokenize input
    inputs = tokenizer(
            case_text,
            truncation=True,
            max_length=512,
            return_tensors='pt',
            padding='max_length'
            ).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.sigmoid(logits).cpu().numpy()[0]

    # Convert to binary predictions based on threshold
    predictions = (probabilities > threshold).astype(int)

    # Prepare detailed results
    results = {
            'probabilities': {article: float(prob) for article, prob in zip(articles, probabilities)},
            'predictions': {article: int(pred) for article, pred in zip(articles, predictions)},
            'predicted_articles': [article for article, pred in zip(articles, predictions) if pred == 1]
            }

    return results

def analyze_prediction(prediction, true_articles):
    """
    Analyze prediction accuracy

    Parameters:
    -----------
    prediction: dict from predict_case_articles
    true_articles: list of true article labels

    Returns:
    --------
    dict: Prediction analysis
    """
    # Get predicted and true articles
    predicted_articles = prediction['predicted_articles']

    # Compute analysis metrics
    correct_predictions = set(predicted_articles) & set(true_articles)
    missed_articles = set(true_articles) - set(predicted_articles)
    false_positives = set(predicted_articles) - set(true_articles)

    return {
            'total_true_articles': len(true_articles),
            'true_articles': list(true_articles),
            'total_predicted_articles': len(predicted_articles),
            'predictions': list(predicted_articles),
            'correct_predictions': list(correct_predictions),
            'missed_articles': list(missed_articles),
            'false_positives': list(false_positives),
            'precision': len(correct_predictions) / len(predicted_articles) if predicted_articles else 0,
            'recall': len(correct_predictions) / len(true_articles) if true_articles else 0
            }

def main():
    # Load trained model and resources
    print("Loading trained model and resources...")
    model, tokenizer, articles = load_trained_model_and_resources()

    # Fetch a random case
    print("\nFetching a random case...")
    case = get_random_case()

    # Print case details
    print(f"\n=== Case ID: {case['case_id']} ===")
    print("\nTrue Articles:", case['true_articles'])

    # Predict articles
    print("\nPredicting articles...")
    prediction = predict_case_articles(model, tokenizer, case['text'], articles)

    # Print prediction probabilities
    print("\nArticle Prediction Probabilities:")
    sorted_probs = sorted(prediction['probabilities'].items(), key=lambda x: x[1], reverse=True)
    for article, prob in sorted_probs:
        print(f"{article}: {prob:.4f}")

    # Analyze prediction
    print("\nAnalyzing predictions...")
    analysis = analyze_prediction(prediction, case['true_articles'])

    # Print detailed analysis
    print("\nDetailed Prediction Analysis:")
    for key, value in analysis.items():
        print(f"{key}: {value}")

if __name__ == "__main__":
    main()
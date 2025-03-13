import argparse
import json
import os
import sqlite3
import torch
import pandas as pd
from transformers import DistilBertTokenizerFast
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

# Set the same constants as in the training script
MAX_TOKEN_COUNT = 512
STRIDE = 256
MODEL_DIR = "/teamspace/studios/this_studio/echr_distilbert_model_sliding_window"
DB_PATH = "/teamspace/studios/this_studio/echr_cases_anonymized.sqlite"


def predict_single_judgment(text, model, tokenizer, article_labels, threshold=0.5):
    """
    Predict articles for a single judgment text

    Args:
        text: The judgment text to analyze
        model: Trained JudgmentsTagger model
        tokenizer: Tokenizer for the model
        article_labels: Mapping between indices and article names
        threshold: Threshold for binary prediction

    Returns:
        List of dictionaries with predicted articles
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # Tokenize with sliding window
    tokenized = tokenizer(
            text,
            truncation=True,
            max_length=MAX_TOKEN_COUNT,
            padding="max_length",
            return_overflowing_tokens=True,
            stride=STRIDE,
            return_tensors="pt",
            )

    # Get number of chunks
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    num_chunks = len(input_ids)

    # Create batch boundaries for a single document
    batch_boundaries = [0, num_chunks]

    # Predict
    with torch.no_grad():
        outputs = model(input_ids, attention_mask, batch_boundaries)

        # Apply sigmoid for probabilities
        probs = torch.sigmoid(outputs).cpu().numpy()[0]  # Get first (only) result

        # Apply threshold
        binary_preds = (probs > threshold).astype(int)

    # Convert to article names with confidence
    predicted_articles = []
    for j, val in enumerate(binary_preds):
        if val == 1:
            predicted_articles.append(
                    {"article": article_labels[str(j)], "confidence": float(probs[j])}
                    )

    # Sort by confidence
    predicted_articles.sort(key=lambda x: x["confidence"], reverse=True)

    return predicted_articles


def load_model_and_tokenizer(model_dir):
    """Load the trained model and tokenizer without needing the class definition"""
    # Load checkpoint without class definition
    checkpoint_path = os.path.join(model_dir, "final_model.ckpt")

    # Load model directly from checkpoint
    model = pl.utilities.cloud_io.load(checkpoint_path)
    model.eval()  # Set to evaluation mode

    # Load tokenizer
    tokenizer = DistilBertTokenizerFast.from_pretrained(model_dir)

    # Load article labels mapping
    with open(os.path.join(model_dir, "article_labels.json"), "r") as f:
        article_labels = json.load(f)

    return model, tokenizer, article_labels


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
        judgment = {"case_id": result[1], "judgement": result[2]}
        # Get associated articles
        cursor.execute(
                f"SELECT normalized_article FROM articles WHERE normalized_article != '' AND case_id = '{judgment['case_id']}';"
                )
        result_articles = cursor.fetchall()
        judgment["articles"] = [item for t in result_articles for item in t]

        return judgment
    except Exception as e:
        print(f"Error fetching judgment: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
            description="Test LinearSVC models against judgments"
            )
    parser.add_argument(
            "--judgment_id",
            type=str,
            help="Specific judgment ID to test (random if not provided)",
            )
    args = parser.parse_args()

    # Load model and tokenizer
    model, tokenizer, article_labels = load_model_and_tokenizer(MODEL_DIR)

    # Get judgment
    judgment = get_judgment(args.judgment_id)
    if not judgment:
        print("No judgment found. Exiting...")
        exit()

    # Predict articles
    predictions = predict_single_judgment(
            judgment["judgment"], model, tokenizer, article_labels
            )

    # Print results
    print("\nPredicted articles:")
    if predictions:
        for article in predictions:
            print(f"  - {article['article']} (confidence: {article['confidence']:.4f})")
    else:
        print("  No articles predicted above threshold")

if __name__ == "__main__":
    main()

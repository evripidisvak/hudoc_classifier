import os
import json
import torch
import sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    DistilBertConfig,
    )
# Add the missing import for PredictionOutput
from transformers.trainer_utils import PredictionOutput
from datasets import Dataset, load_from_disk
import optuna
from torch.profiler import profile, record_function, ProfilerActivity
from torch.amp import autocast, GradScaler
import gc

DB_PATH = "hudoc_classifier/echr_cases_anonymized.sqlite"
MODEL_DIR = "echr_distilbert_model_v2"
DATASETS_DIR = "echr_processed_datasets"
METRICS_FILE = "distilbert_metrics.json"
HYPERPARAMS_FILE = os.path.join(MODEL_DIR, "best_hyperparameters.json")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATASETS_DIR, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set CUDA optimization flags
if torch.cuda.is_available():
    # Enable TF32 for faster matrix multiplications on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # Set benchmark mode to find optimal algorithm for fixed input sizes
    torch.backends.cudnn.benchmark = True

def profile_model_operations(model, trainer, num_steps=5):
    """
    Profile model operations for a few training steps to identify bottlenecks

    Parameters:
    -----------
    model: The PyTorch model to profile
    trainer: HuggingFace Trainer instance
    num_steps: Number of steps to profile (keep small, e.g. 5-10)
    """
    print("\n=== Starting PyTorch Profiler - this may take a moment ===")

    # Get a batch of data
    train_dataloader = trainer.get_train_dataloader()
    batch_iterator = iter(train_dataloader)

    # Move model to the right device if not already
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Create optimizer just for profiling if needed
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Create trace directory
    os.makedirs("profiler_output", exist_ok=True)

    # Start profiling
    with profile(
            activities=[
                    ProfilerActivity.CPU,
                    ProfilerActivity.CUDA,
                    ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_flops=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("profiler_output"),
            ) as prof:
        # Profile a few training steps
        for step in range(num_steps):
            print(f"Profiling step {step+1}/{num_steps}")

            try:
                # Get a batch and move to device
                batch = next(batch_iterator)
            except StopIteration:
                # Restart iterator if needed
                batch_iterator = iter(train_dataloader)
                batch = next(batch_iterator)

            batch = {k: v.to(device) for k, v in batch.items() if isinstance(v, torch.Tensor)}

            # Record data loading
            with record_function("data_loading"):
                # Data is already loaded, this is just to mark the section
                pass

            # Forward pass
            with record_function("forward"):
                outputs = model(**batch)
                loss = outputs.loss

            # Backward pass
            with record_function("backward"):
                loss.backward()

            # Optimizer step
            with record_function("optimizer"):
                optimizer.step()
                optimizer.zero_grad()

            # Let the profiler record the step
            prof.step()

    # Print top 20 operations by CUDA time
    print("\n=== CUDA Time Summary (Top 20) ===")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    # Print top 20 operations by CPU time
    print("\n=== CPU Time Summary (Top 20) ===")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    # Print memory usage
    print("\n=== Memory Usage (Top 10) ===")
    print(prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=10))

    # Clean up to free memory
    gc.collect()

    # Return the profiler for further analysis if needed
    return prof

def fetch_data():
    conn = sqlite3.connect(DB_PATH)
    df_cases = pd.read_sql("SELECT case_id, anonymized_judgement FROM cases", conn)
    df_articles = pd.read_sql(
            "SELECT normalized_article, case_id FROM articles WHERE normalized_article != ''",
            conn,
            )
    conn.close()
    df_cases["anonymized_judgement"] = df_cases["anonymized_judgement"].astype(str)
    df_articles["normalized_article"] = df_articles["normalized_article"].astype(str)
    return df_cases, df_articles


def preprocess_data(df_cases, df_articles):
    # Create a merged dataframe with case text and article labels
    case_texts = df_cases[["case_id", "anonymized_judgement"]].drop_duplicates()

    # Create a pivot table of articles for each case
    # pivot_df = pd.crosstab(
    #         df_articles["case_id"], df_articles["normalized_article"]
    #         ).astype(int)

    pivot_df = pd.crosstab(
            df_articles["case_id"],
            df_articles["normalized_article"],
            values=1,  # Set value to 1 for all matches
            aggfunc='count'  # Count occurrences
            ).fillna(0)
    pivot_df = (pivot_df > 0).astype(int)
    cases_to_articles_df = df_cases.merge(pivot_df, on="case_id", how="left").fillna(0)

    # Merge the text with the article labels
    merged_df = case_texts.merge(pivot_df, on="case_id", how="inner")

    # Get the list of article columns (these will be our classification targets)
    article_columns = pivot_df.columns.tolist()

    # Save article list for future reference
    with open(os.path.join(DATASETS_DIR, "articles.json"), "w") as f:
        json.dump(article_columns, f)

    # Clean up to free memory
    gc.collect()

    return merged_df, article_columns


def compute_class_weights(df, articles):
    """
    Compute class weights to handle imbalanced data.
    For each article, calculate a weight based on the positive/negative ratio.
    """
    weights = []
    for article in articles:
        pos = df[article].sum()
        neg = len(df) - pos
        # Weight is inverse of class frequency, capped at 10x to prevent extreme values
        ratio = neg / pos if pos > 0 else 1.0
        weights.append(min(10.0, ratio))

    print(f"Class weights: {weights}")
    return torch.tensor(weights).to(device)


def create_datasets(df, tokenizer, articles, max_length=512, stride=256, test_size=0.2):
    """
    Create train and test datasets with sliding window for long documents in a memory-efficient way

    Parameters:
    -----------
    df: DataFrame containing case data
    tokenizer: HuggingFace tokenizer
    articles: List of article names to use as labels
    max_length: Maximum sequence length for each window
    stride: Number of overlapping tokens between consecutive windows
    test_size: Fraction of data to use for testing

    Returns:
    --------
    Dictionary containing train and test datasets
    """
    # Reset index to avoid the __index_level_0__ issue
    df = df.reset_index(drop=True)

    # Split data first
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=42)

    # Create empty output directories to save dataset chunks
    train_output_dir = os.path.join(DATASETS_DIR, "train")
    test_output_dir = os.path.join(DATASETS_DIR, "test")
    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    # Process in batches to save memory
    def process_and_save_in_batches(input_df, output_dir, batch_size=1000):
        num_batches = len(input_df) // batch_size + (1 if len(input_df) % batch_size != 0 else 0)
        print(f"Processing {len(input_df)} documents in {num_batches} batches...")

        all_datasets = []

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len(input_df))
            batch_df = input_df.iloc[start_idx:end_idx]

            print(f"Processing batch {batch_idx+1}/{num_batches} ({start_idx}:{end_idx})...")

            # Process each document in the batch
            all_input_ids = []
            all_attention_masks = []
            all_labels = []
            all_case_ids = []
            all_original_indices = []

            for idx, row in batch_df.iterrows():
                text = row["anonymized_judgement"]
                case_id = row["case_id"]

                # Extract labels for this document
                label_values = [float(row[article]) for article in articles]

                # Use tokenizer efficiently for long texts
                # Instead of tokenizing the entire text at once, process it in chunks
                words = text.split()

                # For short documents, process as a single window
                if len(words) <= max_length // 2:  # Approximation
                    encoded = tokenizer(
                            text,
                            truncation=True,
                            max_length=max_length,
                            padding="max_length",
                            return_tensors=None
                            )
                    all_input_ids.append(encoded["input_ids"])
                    all_attention_masks.append(encoded["attention_mask"])
                    all_labels.append(label_values)
                    all_case_ids.append(case_id)
                    all_original_indices.append(idx)
                else:
                    # Split into chunks with stride for long documents
                    chunk_size = max_length // 2  # Approximate number of words per chunk
                    stride_words = stride // 4    # Approximate stride in words

                    for i in range(0, len(words), chunk_size - stride_words):
                        chunk = " ".join(words[i:i + chunk_size])

                        encoded = tokenizer(
                                chunk,
                                truncation=True,
                                max_length=max_length,
                                padding="max_length",
                                return_tensors=None
                                )

                        all_input_ids.append(encoded["input_ids"])
                        all_attention_masks.append(encoded["attention_mask"])
                        all_labels.append(label_values)
                        all_case_ids.append(case_id)
                        all_original_indices.append(idx)

            # Create dataset for this batch
            batch_dict = {
                    "input_ids": all_input_ids,
                    "attention_mask": all_attention_masks,
                    "labels": all_labels,
                    "case_id": all_case_ids,
                    "original_idx": all_original_indices
                    }

            # Convert to Dataset
            batch_dataset = Dataset.from_dict(batch_dict)

            # Save this batch
            batch_path = os.path.join(output_dir, f"batch_{batch_idx}")
            batch_dataset.save_to_disk(batch_path)

            # Clear memory
            del all_input_ids, all_attention_masks, all_labels, all_case_ids, all_original_indices
            del batch_dict, batch_dataset
            gc.collect()

            # Keep track of batch paths
            all_datasets.append(batch_path)

        # After processing all batches, load and concatenate datasets
        print(f"Loading and concatenating {len(all_datasets)} batches...")
        datasets = [load_from_disk(path) for path in all_datasets]

        # Use datasets concatenate method
        combined = concatenate_datasets(datasets)
        combined.save_to_disk(output_dir)

        # Clean up batch files to save disk space
        for path in all_datasets:
            if os.path.exists(path):
                shutil.rmtree(path)

        return combined

    # Import the concatenate_datasets function
    from datasets import concatenate_datasets
    import shutil

    # Process train and test sets
    print("Processing training dataset...")
    train_dataset = process_and_save_in_batches(train_df, train_output_dir)

    print("Processing test dataset...")
    test_dataset = process_and_save_in_batches(test_df, test_output_dir)

    # Final cleanup
    gc.collect()

    return {"train": train_dataset, "test": test_dataset}


def compute_metrics(eval_pred):
    """
    Custom metric function for the Trainer
    Computes F1 scores for each article and a macro F1 score
    """
    predictions, labels = eval_pred
    predictions = (predictions > 0).astype(int)

    results = {}
    # Calculate F1 score for each article
    for i, article in enumerate(articles):
        f1 = f1_score(labels[:, i], predictions[:, i], average="macro", zero_division=0)
        results[f"f1_{article}"] = f1

    # Calculate overall macro F1 score
    results["macro_f1"] = f1_score(
            labels.flatten(), predictions.flatten(), average="macro", zero_division=0
            )
    return results


class MultiLabelClassificationTrainer(Trainer):
    """
    Custom Trainer to handle class weights for multi-label classification
    and aggregate predictions from multiple windows of the same document
    """

    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(device) if class_weights is not None else None
        self.scaler = GradScaler() if torch.cuda.is_available() and self.args.fp16 else None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        inputs = {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        labels = inputs.pop("labels")
        # Use autocast for mixed precision
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        with autocast(device_type=device_type, enabled=self.args.fp16):
            outputs = model(**inputs)
            logits = outputs.logits

            # Basic BCEWithLogitsLoss
            loss_fct = torch.nn.BCEWithLogitsLoss(
                    weight=self.class_weights, reduction="mean"
                    )
            loss = loss_fct(
                    logits.view(-1, self.model.config.num_labels),
                    labels.float().view(-1, self.model.config.num_labels),
                    )

        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        """
        Override prediction_step to keep track of original_idx for window aggregation
        """
        has_original_idx = "original_idx" in inputs
        original_idx = inputs.pop("original_idx", None)

        # Call the parent class's prediction_step
        loss, logits, labels = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        # If we're in prediction mode and have original_idx, return it alongside the predictions
        if not prediction_loss_only and has_original_idx:
            return loss, logits, labels, original_idx

        return loss, logits, labels

    def predict(self, test_dataset, ignore_keys=None, metric_key_prefix="test"):
        """
        Override predict to aggregate predictions from multiple windows of the same document
        """
        # Run the standard prediction step to get window-level predictions
        outputs = super().predict(test_dataset, ignore_keys, metric_key_prefix)

        # If the dataset doesn't have original_idx, return standard predictions
        if not hasattr(test_dataset, "features") or "original_idx" not in test_dataset.features:
            return outputs

        # Get the original indices and predictions
        original_indices = np.array(test_dataset["original_idx"])
        window_predictions = outputs.predictions

        # Find unique document indices
        unique_docs = np.unique(original_indices)
        num_docs = len(unique_docs)
        num_labels = window_predictions.shape[1]

        # Prepare aggregated predictions
        doc_predictions = np.zeros((num_docs, num_labels))

        # Aggregate predictions for each document using max pooling across windows
        for i, doc_idx in enumerate(unique_docs):
            # Get all windows belonging to this document
            doc_windows = original_indices == doc_idx
            # Max pooling: take highest prediction score across all windows
            doc_predictions[i] = np.max(window_predictions[doc_windows], axis=0)

        # Create a new PredictionOutput with aggregated predictions
        aggregated_outputs = PredictionOutput(
                predictions=doc_predictions,
                label_ids=outputs.label_ids[np.array([np.where(original_indices == idx)[0][0] for idx in unique_docs])],
                metrics=outputs.metrics,
                )

        return aggregated_outputs

def load_or_tune_hyperparameters(
        train_dataset,
        eval_dataset,
        articles,
        tokenizer,
        class_weights=None,
        n_trials=10,
        force_new=False,
        ):
    """
    Load saved hyperparameters from file or run a new hyperparameter search if needed

    Parameters:
    -----------
    force_new: bool
        If True, run a new hyperparameter search even if saved hyperparameters exist
    """
    print("=== === === === Starting hyperparameter tuning === === === ===")

    # Check if we have saved hyperparameters
    if os.path.exists(HYPERPARAMS_FILE) and not force_new:
        print("Loading saved hyperparameters...")
        with open(HYPERPARAMS_FILE, "r") as f:
            return json.load(f)

    print("Running hyperparameter optimization with Optuna...")

    tuning_train_dataset = train_dataset.select(range(min(1000, len(train_dataset))))
    tuning_eval_dataset = eval_dataset.select(range(min(500, len(eval_dataset))))

    def model_init():
        return DistilBertForSequenceClassification.from_pretrained(
                "distilbert-base-uncased",
                num_labels=len(articles),
                problem_type="multi_label_classification",
                )

    training_args = TrainingArguments(
            output_dir=f"{MODEL_DIR}/hp_tuning",
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=2,  # Shorter for hyperparameter tuning
            logging_dir=f"{MODEL_DIR}/hp_logs",
            logging_steps=50,
            fp16=True,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            )

    data_collator = DataCollatorWithPadding(tokenizer)

    trainer = MultiLabelClassificationTrainer(
            model_init=model_init,
            args=training_args,
            train_dataset=tuning_train_dataset,
            eval_dataset=tuning_eval_dataset,
            compute_metrics=compute_metrics,
            class_weights=class_weights,
            data_collator=data_collator,
            )

    def hp_space(trial):
        return {
                "learning_rate": trial.suggest_float("learning_rate", 1e-5, 5e-5, log=True),
                "per_device_train_batch_size": trial.suggest_categorical(
                        "per_device_train_batch_size", [8, 16, 24, 32]
                        ),
                "per_device_eval_batch_size": trial.suggest_categorical(
                        "per_device_eval_batch_size", [8, 16, 24, 32]
                        ),
                "gradient_accumulation_steps": trial.suggest_categorical(
                        "gradient_accumulation_steps", [1, 4, 8]
                        ),
                "warmup_ratio": trial.suggest_float("warmup_ratio", 0.05, 0.3),
                "weight_decay": trial.suggest_float("weight_decay", 0.01, 0.1),
                }

    best_trial = trainer.hyperparameter_search(
            direction="maximize",
            backend="optuna",
            hp_space=hp_space,
            n_trials=n_trials,
            sampler=optuna.samplers.TPESampler(n_startup_trials=1),
            compute_objective=lambda metrics: metrics["eval_macro_f1"],
            pruner=optuna.pruners.MedianPruner(n_startup_trials=1),
            )

    print(f"Best hyperparameters: {best_trial.hyperparameters}")

    # Save the best hyperparameters
    os.makedirs(os.path.dirname(HYPERPARAMS_FILE), exist_ok=True)
    with open(HYPERPARAMS_FILE, "w") as f:
        json.dump(best_trial.hyperparameters, f, indent=4)

    # Clear memory after hyperparameter tuning
    torch.cuda.empty_cache()
    gc.collect()

    print("=== === === === Hyperparameter tuning finished === === === ===")
    return best_trial.hyperparameters


def train_model_with_hyperparams(
        train_dataset, eval_dataset, articles, tokenizer, hyperparams, class_weights=None
        ):
    print("=== === === === Starting training === === === ===")
    """Train model with specific hyperparameters"""
    num_labels = len(articles)

    # Check if a saved model exists
    # if os.path.exists(MODEL_DIR):
    #     print("Existing model found. Loading for continued training...")
    #     model = DistilBertForSequenceClassification.from_pretrained(
    #             MODEL_DIR,
    #             num_labels=num_labels,
    #             problem_type="multi_label_classification",
    #             )
    #     print("Loaded existing model weights.")
    # else:
    # Create a new model if no existing model is found
    config = DistilBertConfig.from_pretrained(
            "distilbert-base-uncased",
            num_labels=num_labels,
            problem_type="multi_label_classification",
            attention_dropout=0.1,
            hidden_dropout_prob=0.1,
            )

    # Create the multi-label classification model
    model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            config=config,
            )
    print("Created new model from scratch.")

    model.gradient_checkpointing_enable()
    model.to(device)

    # Set up training arguments using the best hyperparameters
    training_args = TrainingArguments(
            output_dir=MODEL_DIR,
            eval_strategy="epoch",
            save_strategy="epoch",
            num_train_epochs=20,
            per_device_train_batch_size=hyperparams.get("per_device_train_batch_size", 8),
            gradient_accumulation_steps=hyperparams.get("gradient_accumulation_steps", 4),
            per_device_eval_batch_size=hyperparams.get("per_device_eval_batch_size", 8),
            logging_dir=f"{MODEL_DIR}/logs",
            logging_strategy="steps",
            logging_steps=500,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="macro_f1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            warmup_ratio=hyperparams.get("warmup_ratio", 0.1),
            weight_decay=hyperparams.get("weight_decay", 0.01),
            learning_rate=hyperparams.get("learning_rate", 2e-5),
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            gradient_checkpointing=True,
            optim="adamw_torch",
            # bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
            no_cuda=not torch.cuda.is_available(),
            ddp_find_unused_parameters=False,
            )

    data_collator = DataCollatorWithPadding(tokenizer)

    # Use custom trainer with class weights
    trainer = MultiLabelClassificationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            class_weights=class_weights,
            )

    # Add early stopping callback
    trainer.add_callback(
            EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.001)
            )

    # profile_results = profile_model_operations(model, trainer, num_steps=5)

    trainer.train()
    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()

    print("=== === === === Training finished === === === ===")
    return model, trainer


def evaluate_model(model, trainer, eval_dataset, articles):
    """
    Evaluate the model with support for the sliding window approach
    """
    # Make predictions with window aggregation
    outputs = trainer.predict(eval_dataset)
    predictions = outputs.predictions

    # Convert to binary predictions (threshold = 0.5)
    preds = (predictions > 0).astype(int)

    # Get true labels - we need to handle the case where we have multiple windows per document
    if "original_idx" in eval_dataset.features:
        # Get unique document indices
        unique_docs = np.unique(eval_dataset["original_idx"])
        # Get the first window's labels for each document (all windows have the same labels)
        true_labels = np.array([
                eval_dataset["labels"][np.where(np.array(eval_dataset["original_idx"]) == doc_idx)[0][0]]
                for doc_idx in unique_docs
                ])
    else:
        # Standard case - use all labels
        true_labels = np.array(eval_dataset["labels"])

    # Calculate metrics for each article
    metrics = {}
    for i, article in enumerate(articles):
        article_metrics = classification_report(
                true_labels[:, i], preds[:, i], output_dict=True, zero_division=0
                )
        metrics[article] = article_metrics

    # Also calculate macro averages across all articles
    metrics["macro_avg"] = classification_report(
            true_labels.flatten(), preds.flatten(), output_dict=True, zero_division=0
            )

    # Save metrics
    with open(os.path.join(MODEL_DIR, METRICS_FILE), "w") as f:
        json.dump(metrics, f, indent=4)

    return metrics


def load_articles():
    """Load the list of articles if it exists"""
    articles_path = os.path.join(DATASETS_DIR, "articles.json")
    if os.path.exists(articles_path):
        with open(articles_path, "r") as f:
            return json.load(f)
    return None


if __name__ == "__main__":
    torch.cuda.empty_cache()
    gc.collect()
    import argparse

    # Create command line arguments
    parser = argparse.ArgumentParser(description="Train ECHR article classifier")
    parser.add_argument(
            "--retune", action="store_true", help="Force re-tuning of hyperparameters"
            )
    parser.add_argument(
            "--trials", type=int, default=5, help="Number of hyperparameter trials"
            )
    args = parser.parse_args()

    # Clear memory
    torch.cuda.empty_cache()
    gc.collect()

    # Check if we can load articles and datasets directly
    articles = load_articles()

    # Check if datasets already exist
    if (
            articles is not None
            and os.path.exists(os.path.join(DATASETS_DIR, "train"))
            and os.path.exists(os.path.join(DATASETS_DIR, "test"))
    ):
        print(f"Loading preprocessed data with {len(articles)} articles...")
        datasets = {
                "train": load_from_disk(os.path.join(DATASETS_DIR, "train")),
                "test": load_from_disk(os.path.join(DATASETS_DIR, "test")),
                }

        # For class weights, we need the original dataframe
        print("Fetching data for class weight calculation...")
        df_cases, df_articles = fetch_data()
        df, _ = preprocess_data(df_cases, df_articles)
        class_weights = compute_class_weights(df, articles)
    else:
        print("Fetching and preprocessing data...")
        df_cases, df_articles = fetch_data()
        df, articles = preprocess_data(df_cases, df_articles)
        print(f"Found {len(articles)} unique ECHR articles")

        # Calculate class weights from the full dataset
        class_weights = compute_class_weights(df, articles)

        print("Loading tokenizer...")
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        print("Creating datasets...")
        datasets = create_datasets(df, tokenizer, articles)

    print(f"Train set: {len(datasets['train'])} examples")
    print(f"Test set: {len(datasets['test'])} examples")

    # Load tokenizer (either new or from saved model)
    if os.path.exists(os.path.join(MODEL_DIR, "tokenizer_config.json")):
        tokenizer = DistilBertTokenizer.from_pretrained(MODEL_DIR)
    else:
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Get hyperparameters - load from file or perform tuning
    best_hyperparams = load_or_tune_hyperparameters(
            datasets["train"],
            datasets["test"],
            articles,
            tokenizer,
            class_weights=class_weights,
            n_trials=args.trials,
            force_new=args.retune,
            )

    print(f"Using hyperparameters: {best_hyperparams}")

    # Train model with best hyperparameters
    print("Training model with optimized hyperparameters...")
    model, trainer = train_model_with_hyperparams(
            datasets["train"],
            datasets["test"],
            articles,
            tokenizer,
            best_hyperparams,
            class_weights,
            )

    print("Evaluating model...")
    metrics = evaluate_model(model, trainer, datasets["test"], articles)

    print("Training complete. Metrics saved to", os.path.join(MODEL_DIR, METRICS_FILE))
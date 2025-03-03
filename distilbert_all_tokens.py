import os
import re
import sqlite3
import nltk
import json
import gc

print("Defining parameters...")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:512"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "False"

from torch.utils.checkpoint import checkpoint
import torch.distributed as dist
import deepspeed
from pathlib import Path
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks import DeviceStatsMonitor, StochasticWeightAveraging
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profilers import PyTorchProfiler
from pytorch_lightning.tuner.tuning import Tuner
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchmetrics.functional import accuracy, auroc, f1_score, precision, recall
from transformers import (
    DistilBertModel,
    DistilBertTokenizerFast as DistilBertTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorWithPadding,
    )

RANDOM_SEED = 42
MAX_TOKEN_COUNT = 512  # DistilBERT's maximum token length
CHUNK_OVERLAP = 50  # Number of overlapping tokens between chunks
N_EPOCHS = 10
N_ACCUMULATE_BATCHES = 2
BATCH_SIZE = 32
GRADIENT_ACCUMULATION_STEPS = 2
EVAL_FREQUENCY = 1
HIDDEN_DROPOUT_PROB = 0.3
ATTENTION_PROBS_DROPOUT_PROB = 0.3
BERT_MODEL_NAME = "distilbert-base-uncased"
DB_PATH = "/teamspace/studios/this_studio/echr_cases_anonymized.sqlite"

model_name = "echr_judgments_classifier"
model_dir = "/teamspace/studios/this_studio/echr_distilbert_model"
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
LR_CONFIG = model_dir + "/lr_config.json"

pl.seed_everything(RANDOM_SEED)
nltk.download("stopwords")
nltk.download("wordnet")
torch.set_float32_matmul_precision("medium")
os.makedirs(model_dir, exist_ok=True)

torch.set_float32_matmul_precision("medium")
torch.backends.cuda.matmul.allow_tf32 = True


class JudgmentsDataset(Dataset):
    def __init__(
            self,
            data: pd.DataFrame,
            tokenizer: DistilBertTokenizer,
            max_token_len: int = MAX_TOKEN_COUNT,
            chunk_overlap: int = CHUNK_OVERLAP
            ):
        self.tokenizer = tokenizer
        self.data = data[['judgment'] + ARTICLES_COLUMNS].copy()
        self.max_token_len = max_token_len
        self.chunk_overlap = chunk_overlap

    def __len__(self):
        return len(self.data)

    def create_chunks(self, text):
        # Tokenize the entire text
        tokens = self.tokenizer.tokenize(text)

        # Calculate chunk size (leaving room for special tokens)
        chunk_size = self.max_token_len - 2  # Account for [CLS] and [SEP]

        # If text is shorter than max length, return it as a single chunk
        if len(tokens) <= chunk_size:
            return [text]

        chunks = []
        start = 0
        while start < len(tokens):
            # Get chunk of tokens
            chunk_tokens = tokens[start:start + chunk_size]
            # Convert tokens back to text
            chunk_text = self.tokenizer.convert_tokens_to_string(chunk_tokens)
            chunks.append(chunk_text)
            # Move start pointer, accounting for overlap
            start += (chunk_size - self.chunk_overlap)

        return chunks

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        judgment = data_row.judgment
        labels = data_row[ARTICLES_COLUMNS]
        labels = labels.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Split text into chunks
        chunks = self.create_chunks(judgment)

        # Tokenize all chunks
        encodings = [
                self.tokenizer(
                        chunk,
                        max_length=self.max_token_len,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                        ) for chunk in chunks
                ]

        # Prepare the final tensors
        input_ids = torch.cat([encoding["input_ids"] for encoding in encodings])
        attention_mask = torch.cat([encoding["attention_mask"] for encoding in encodings])

        # Create labels tensor (same labels for all chunks)
        labels_tensor = torch.tensor(labels.values.astype(np.float32), dtype=torch.float32)
        labels_repeated = labels_tensor.unsqueeze(0).repeat(len(chunks), 1)

        return {
                "input_ids": input_ids.squeeze(),
                "attention_mask": attention_mask.squeeze(),
                "labels": labels_repeated,
                "chunk_count": len(chunks)
                }

class ChunkCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # Separate the chunks and their metadata
        max_chunks = max(item["chunk_count"] for item in batch)

        # Initialize tensors for the batch
        batch_size = len(batch)
        input_ids = torch.zeros(batch_size, max_chunks, MAX_TOKEN_COUNT, dtype=torch.long)
        attention_mask = torch.zeros(batch_size, max_chunks, MAX_TOKEN_COUNT, dtype=torch.long)
        labels = torch.zeros(batch_size, max_chunks, len(ARTICLES_COLUMNS), dtype=torch.float)
        chunk_counts = torch.tensor([item["chunk_count"] for item in batch])

        # Fill the tensors
        for i, item in enumerate(batch):
            chunks = item["chunk_count"]
            input_ids[i, :chunks] = item["input_ids"]
            attention_mask[i, :chunks] = item["attention_mask"]
            labels[i, :chunks] = item["labels"]

        return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "chunk_counts": chunk_counts
                }

class JudgmentsDataModule(pl.LightningDataModule):
    def __init__(
            self,
            train_df,
            test_df,
            tokenizer,
            batch_size=32,
            max_token_len=MAX_TOKEN_COUNT,
            chunk_overlap=CHUNK_OVERLAP
            ):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.chunk_overlap = chunk_overlap
        self.collate_fn = ChunkCollator(tokenizer)

    def setup(self, stage=None):
        self.train_dataset = JudgmentsDataset(
                self.train_df,
                self.tokenizer,
                self.max_token_len,
                self.chunk_overlap
                )

        self.test_dataset = JudgmentsDataset(
                self.test_df,
                self.tokenizer,
                self.max_token_len,
                self.chunk_overlap
                )

    def train_dataloader(self):
        return DataLoader(
                self.train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=8,
                pin_memory=True,
                collate_fn=self.collate_fn,
                prefetch_factor=4,
                persistent_workers=True,
                )

    def val_dataloader(self):
        return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=4,
                collate_fn=self.collate_fn,
                )

    def test_dataloader(self):
        return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                num_workers=4,
                collate_fn=self.collate_fn,
                )


class JudgmentsTagger(pl.LightningModule):
    def __init__(
            self, n_classes: int, batch_size=32, learning_rate=0.001, class_weights=None
            ):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.bert = DistilBertModel.from_pretrained(
                BERT_MODEL_NAME,
                return_dict=True,
                )

        self.bert.gradient_checkpointing_enable()

        self.classifier = nn.Sequential(
                nn.Dropout(ATTENTION_PROBS_DROPOUT_PROB),
                nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size // 2),
                nn.GELU(),
                nn.Dropout(ATTENTION_PROBS_DROPOUT_PROB),
                nn.Linear(self.bert.config.hidden_size // 2, n_classes)
                )

        # Add a final aggregation layer
        self.chunk_aggregator = nn.Sequential(
                nn.Linear(n_classes * 2, n_classes),  # *2 for concatenating max and mean
                nn.GELU(),
                nn.Dropout(ATTENTION_PROBS_DROPOUT_PROB),
                nn.Linear(n_classes, n_classes)
                )

        self.n_training_steps = None
        self.n_warmup_steps = None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    def forward(self, input_ids, attention_mask, chunk_counts=None):
        batch_size = input_ids.size(0)
        max_chunks = input_ids.size(1)

        # Reshape for processing all chunks
        input_ids = input_ids.view(-1, input_ids.size(-1))
        attention_mask = attention_mask.view(-1, attention_mask.size(-1))

        def bert_forward():
            return self.bert(input_ids, attention_mask=attention_mask)

        with torch.cuda.amp.autocast():
            bert_output = checkpoint(bert_forward)
            sequence_output = bert_output.last_hidden_state
            pooled_output = sequence_output[:, 0]  # Use [CLS] token

            # Get predictions for all chunks
            chunk_logits = self.classifier(pooled_output)
            chunk_logits = chunk_logits.view(batch_size, max_chunks, -1)

            # Create mask for valid chunks
            if chunk_counts is not None:
                chunk_mask = torch.arange(max_chunks, device=chunk_counts.device).unsqueeze(0) < chunk_counts.unsqueeze(1)
                chunk_mask = chunk_mask.unsqueeze(-1).expand(-1, -1, chunk_logits.size(-1))
                chunk_logits = chunk_logits.masked_fill(~chunk_mask, float('-inf'))

            # Get both max and mean of chunk predictions
            max_logits = torch.max(chunk_logits, dim=1)[0]

            # For mean, replace -inf with 0 before averaging
            mean_mask = (chunk_logits != float('-inf')).float()
            masked_logits = chunk_logits.masked_fill(chunk_logits == float('-inf'), 0.0)
            mean_logits = (masked_logits.sum(dim=1) / (mean_mask.sum(dim=1) + 1e-10))

            # Concatenate max and mean logits
            combined_logits = torch.cat([max_logits, mean_logits], dim=-1)

            # Final prediction using the aggregator
            final_logits = self.chunk_aggregator(combined_logits)
            output = torch.sigmoid(final_logits)

            # Store chunk predictions for analysis if needed
            self.last_chunk_predictions = torch.sigmoid(chunk_logits)

        return output, self.last_chunk_predictions

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"][:, 0]  # Take first chunk's labels (they're all the same)
        chunk_counts = batch["chunk_counts"]

        outputs, chunk_predictions = self(input_ids, attention_mask, chunk_counts)

        # Calculate main loss on final predictions
        main_loss = self.criterion(outputs, labels.float())

        # Calculate auxiliary loss on individual chunk predictions
        chunk_labels = labels.unsqueeze(1).expand(-1, chunk_predictions.size(1), -1)
        chunk_mask = torch.arange(chunk_predictions.size(1), device=chunk_counts.device).unsqueeze(0) < chunk_counts.unsqueeze(1)
        # Expand mask to match the predictions dimensions
        chunk_mask = chunk_mask.unsqueeze(-1).expand(-1, -1, chunk_predictions.size(-1))

        chunk_loss = self.criterion(
                chunk_predictions[chunk_mask].view(-1, chunk_predictions.size(-1)),
                chunk_labels[chunk_mask].view(-1, chunk_labels.size(-1))
                )

        # Combine losses (give more weight to main loss)
        total_loss = 0.7 * main_loss + 0.3 * chunk_loss

        self.log("train_loss", total_loss, prog_bar=True, logger=True)
        self.log("train_main_loss", main_loss, prog_bar=False, logger=True)
        self.log("train_chunk_loss", chunk_loss, prog_bar=False, logger=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"][:, 0]
        chunk_counts = batch["chunk_counts"]

        outputs, chunk_predictions = self(input_ids, attention_mask, chunk_counts)

        main_loss = self.criterion(outputs, labels.float())

        # Calculate chunk-level metrics
        chunk_labels = labels.unsqueeze(1).expand(-1, chunk_predictions.size(1), -1)
        chunk_mask = torch.arange(chunk_predictions.size(1), device=chunk_counts.device).unsqueeze(0) < chunk_counts.unsqueeze(1)
        # Expand mask to match the predictions dimensions
        chunk_mask = chunk_mask.unsqueeze(-1).expand(-1, -1, chunk_predictions.size(-1))

        chunk_loss = self.criterion(
                chunk_predictions[chunk_mask].view(-1, chunk_predictions.size(-1)),
                chunk_labels[chunk_mask].view(-1, chunk_labels.size(-1))
                )

        total_loss = 0.7 * main_loss + 0.3 * chunk_loss

        self.log("val_loss", total_loss, prog_bar=True, logger=True)
        self.log("val_main_loss", main_loss, prog_bar=False, logger=True)
        self.log("val_chunk_loss", chunk_loss, prog_bar=False, logger=True)

        return total_loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"][:, 0]
        chunk_counts = batch["chunk_counts"]

        outputs, chunk_predictions = self(input_ids, attention_mask, chunk_counts)

        # Calculate main metrics
        main_metrics = self.calculate_metrics(outputs, labels)

        # Calculate chunk-level metrics
        chunk_labels = labels.unsqueeze(1).expand(-1, chunk_predictions.size(1), -1)
        chunk_mask = torch.arange(chunk_predictions.size(1), device=chunk_counts.device).unsqueeze(0) < chunk_counts.unsqueeze(1)
        chunk_mask = chunk_mask.unsqueeze(-1)

        # Get metrics for each chunk
        chunk_metrics = []
        for i in range(chunk_predictions.size(1)):
            chunk_pred = chunk_predictions[:, i][chunk_mask[:, i, 0]]
            chunk_lab = chunk_labels[:, i][chunk_mask[:, i, 0]]
            if len(chunk_pred) > 0:  # Only calculate metrics if we have predictions for this chunk
                chunk_metrics.append(self.calculate_metrics(chunk_pred, chunk_lab))

        return {
                "test_metrics": main_metrics,
                "chunk_metrics": chunk_metrics,
                "chunk_predictions": chunk_predictions[chunk_mask].cpu().numpy(),
                "chunk_counts": chunk_counts.cpu().numpy()
                }

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)

        if self.trainer and self.trainer.datamodule:
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            self.n_training_steps = steps_per_epoch * self.trainer.max_epochs
            self.n_warmup_steps = int(self.n_training_steps * 0.10)
        else:
            raise ValueError("Trainer or datamodule not initialized")

        scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.n_warmup_steps,
                num_training_steps=self.n_training_steps
                )

        return {
                "optimizer": optimizer,
                "lr_scheduler": {
                        "scheduler": scheduler,
                        "interval": "step",
                        "frequency": 1,
                        "monitor": "val_loss",
                        "strict": True,
                        "name": None,
                        },
                }

def calculate_multilabel_class_weights(y: np.ndarray):
    num_labels = y.shape[1]
    class_weights = []

    for i in range(num_labels):
        # Flatten the current label to a 1D array for computation
        label_targets = y[:, i]

        # Calculate class weight for the current label
        unique_classes = np.unique(label_targets)

        print(f"Label {i} unique classes: {unique_classes}")

        if len(unique_classes) == 1:
            # If only 0s or only 1s, set weight to 1.0
            class_weights.append(1.0)
        else:
            weights = compute_class_weight(
                    class_weight="balanced", classes=unique_classes, y=label_targets
                    )
            # Weight for the positive class (label=1)
            class_weights.append(weights[1])

    return np.array(class_weights)


def fetch_cases():
    try:
        print("Connecting to DB...")
        # Connect to DB and create a cursor
        sqlite_connection = sqlite3.connect(DB_PATH)
        cursor = sqlite_connection.cursor()
        print("DB connection successful")

        print("Fetch cases...")
        # Write a query and execute it with cursor
        # query_cases = "select case_id, anonymized_judgement from cases limit 200;"
        query_cases = "select case_id, anonymized_judgement from cases;"
        cursor.execute(query_cases)

        # Fetch result
        result_cases = cursor.fetchall()

        print("Fetch articles...")
        query_articles = "select normalized_article, case_id from articles where article != '';"
        # query_articles = "select normalized_article, case_id from articles where article != '' GROUP by case_id, normalized_article;"
        cursor.execute(query_articles)

        # Fetch result
        result_articles = cursor.fetchall()

        # Close the cursor
        cursor.close()

        # Create Dataframes
        df_cases = pd.DataFrame(result_cases, columns=["case_id", "judgment"])
        df_articles = pd.DataFrame(result_articles, columns=["article", "case_id"])

        return [df_cases, df_articles]

    # Handle errors
    except sqlite3.Error as error:
        print("Error occurred - ", error)
        exit()

    # Close DB Connection irrespective of success or failure
    finally:
        if sqlite_connection:
            sqlite_connection.close()
            print("SQLite Connection closed")


def find_min_max_tokens(df, tokenizer):
    print("Calculating min / max tokens")
    # Tokenize judgments and calculate their lengths
    token_lengths = df["judgment"].apply(lambda x: len(tokenizer.tokenize(x)))
    print(f"Total tokens: {len(token_lengths)}")

    # Find min and max lengths
    min_tokens = token_lengths.min()
    max_tokens = token_lengths.max()

    return min_tokens, max_tokens


def save_learning_rate(lr_value, save_path=LR_CONFIG):
    """
    Save the optimal learning rate to a JSON file.

    Parameters:
    lr_value (float): The learning rate value to save
    save_path (str): Path to save the JSON file
    """
    config = {"learning_rate": lr_value}
    with open(save_path, "w") as f:
        json.dump(config, f)
    print(f"Saved learning rate {lr_value} to {save_path}")


def load_learning_rate(load_path=LR_CONFIG, default_lr=1e-5):
    """
    Load the learning rate from a JSON file if it exists.

    Parameters:
    load_path (str): Path to load the JSON file from
    default_lr (float): Default learning rate to use if file doesn't exist

    Returns:
    float: The loaded learning rate or default value
    """
    if os.path.exists(load_path):
        with open(load_path, "r") as f:
            config = json.load(f)
            lr = config.get("learning_rate", default_lr)
        print(f"Loaded learning rate {lr} from {load_path}")
        return lr
    print(f"No saved learning rate found at {load_path}")
    return None

def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

if __name__ == "__main__":
    df_cases, df_articles = fetch_cases()

    print("Cases head")
    print(df_cases.head())
    print("Cases shape: ", df_cases.shape)

    print("Articles head")
    print(df_articles.head())
    print("Articles shape: ", df_articles.shape)

    # Preprocess and clean up data
    print("Preprocessing data...")
    articles = list(df_articles.head().article.unique())
    articles = df_articles.article.unique().tolist()

    # Match cases to articles
    merged_df = df_cases.merge(
            df_articles, left_on="case_id", right_on="case_id", how="left"
            )
    pivot_df = pd.crosstab(merged_df["case_id"], merged_df["article"])
    cases_to_articles_df = df_cases.merge(
            pivot_df, left_on="case_id", right_index=True, how="left"
            ).fillna(0)

    # Total plots
    ARTICLES_COLUMNS = cases_to_articles_df.columns.tolist()[2:]
    print("Unique articles in the selected cases: ", len(ARTICLES_COLUMNS))
    print("Articles: ", ARTICLES_COLUMNS)

    # Calculate the number of articles linked to each case
    cases_to_articles_df["num_articles"] = cases_to_articles_df[ARTICLES_COLUMNS].sum(
            axis=1
            )

    # Remove cases with no linked articles
    cases_to_articles_df.drop(
            cases_to_articles_df[cases_to_articles_df.num_articles <= 0].index, inplace=True
            )

    # Get a breakdown of the counts
    article_counts_breakdown = (
            cases_to_articles_df["num_articles"].value_counts().sort_index()
    )

    # Print the breakdown
    print("Breakdown of the number of articles each case is linked to:")
    print(article_counts_breakdown)

    cases_to_articles_df.drop("num_articles", axis=1, inplace=True)

    print("Processed cases head")
    print(cases_to_articles_df.head())

    train_df, val_df = train_test_split(cases_to_articles_df, test_size=0.2)
    print("Training shape: ", train_df.shape)
    print("Validation shape: ", val_df.shape)

    label_counts = train_df[ARTICLES_COLUMNS].sum(axis=0)
    print("Positive label counts per class:")
    print(label_counts)

    # Get the true labels for your training data
    train_labels = train_df[ARTICLES_COLUMNS].values
    print("Train labels shape: ", train_labels.shape)
    print("Train labels: ", train_labels)
    print("unique ", np.unique(train_labels))

    # Calculate weights
    class_weights = calculate_multilabel_class_weights(train_labels)
    print("Class weights: ", class_weights)

    # Convert weights to a PyTorch tensor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    print(class_weights)

    tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_NAME)
    bert_model = DistilBertModel.from_pretrained(BERT_MODEL_NAME)
    bert_model = torch.compile(bert_model)
    bert_model = bert_model.to(memory_format=torch.channels_last)

    # Try on a random row
    sample_row = cases_to_articles_df.sample(1).squeeze()
    sample_case_id = sample_row.case_id
    sample_judgment = sample_row.judgment
    sample_articles = sample_row[ARTICLES_COLUMNS]
    d = sample_articles.to_dict()
    sample_articles = {
            article: match
            for article, match in sample_articles.to_dict().items()
            if match > 0
            }
    print("Sample case id: ", sample_case_id)
    print("Sample case articles: ", sample_articles)

    # todo do I need this?
    encoding = tokenizer.encode_plus(
            sample_judgment,
            add_special_tokens=True,
            max_length=MAX_TOKEN_COUNT,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            )

    tokenizer.save_pretrained(model_dir)

    encoding.keys()
    print("input_ids shape: ", encoding["input_ids"].shape)
    print("attention_mask shape: ", encoding["attention_mask"].shape)

    train_dataset = JudgmentsDataset(train_df, tokenizer, max_token_len=MAX_TOKEN_COUNT)

    print("data_module")
    data_module = JudgmentsDataModule(
            train_df,
            val_df,
            tokenizer,
            batch_size=BATCH_SIZE,
            max_token_len=MAX_TOKEN_COUNT,
            )

    print("model")
    model = JudgmentsTagger(n_classes=len(ARTICLES_COLUMNS))

    print("checkpoint")
    checkpoint_callback = ModelCheckpoint(
            dirpath=model_dir,
            filename=model_name,
            save_top_k=1,
            verbose=True,
            monitor="val_loss",
            mode="min",
            )

    print("logger")
    logger = TensorBoardLogger("lightning_logs", name="judgments")
    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=2)

    # print("profiler")
    # profiler = pl.profilers.SimpleProfiler()

    print("trainer")
    # precision="bf16-mixed",
    trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            precision="16-mixed",
            accumulate_grad_batches=GRADIENT_ACCUMULATION_STEPS,
            max_epochs=N_EPOCHS,
            strategy="deepspeed_stage_2",
            num_sanity_val_steps=0,
            check_val_every_n_epoch=EVAL_FREQUENCY,
            gradient_clip_val=0.5,
            logger=logger,
            callbacks=[
                    early_stopping_callback,
                    checkpoint_callback,
                    DeviceStatsMonitor()
                    ],
            enable_checkpointing=True,
            enable_progress_bar=True,
            deterministic=True,
            log_every_n_steps=50
            )

    print("Tuner")
    # tuner = Tuner(trainer)

    # First check if we have a saved learning rate
    saved_lr = load_learning_rate()

    if saved_lr is None:
        print("Finding optimal learning rate...")

        batch_size_finder_trainer = pl.Trainer(
                accelerator="gpu",
                devices=1,
                precision="16-mixed",
                max_epochs=1,  # Only needed for tuning
                logger=False,   # Avoid unnecessary logs
                enable_checkpointing=False,
                enable_progress_bar=False
                )
        tuner = Tuner(batch_size_finder_trainer)
        # Scale batch size first as it may affect optimal learning rate
        print("Scaling batch size...")
        tuner.scale_batch_size(
                model=model, mode="binsearch", datamodule=data_module, init_val=5
                )

        print("Running learning rate finder...")
        lr_finder = tuner.lr_find(model=model, datamodule=data_module)

        # Get the suggestion for the learning rate
        suggested_lr = lr_finder.suggestion()

        # Save the found learning rate
        save_learning_rate(suggested_lr)

        # Update the model's learning rate
        model.learning_rate = suggested_lr
    else:
        print(f"Using saved learning rate: {saved_lr}")
        model.learning_rate = saved_lr

    print(f"Training with learning rate: {model.learning_rate}")

    print(BERT_MODEL_NAME)
    print("fit")
    cleanup()
    trainer.fit(model=model, datamodule=data_module)

    print("vaidate")
    trainer.validate(model=model, datamodule=data_module)
    print("test")
    trainer.test(ckpt_path="best", datamodule=data_module)

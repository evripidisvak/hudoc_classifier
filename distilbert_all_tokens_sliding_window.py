import gc
import json
import os
import sqlite3

import nltk

print("Defining parameters...")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.6,max_split_size_mb:512"
os.environ["TOKENIZERS_PARALLELISM"] = "False"

import torch
from torch.utils.checkpoint import checkpoint
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import (DistilBertModel, DistilBertTokenizerFast as DistilBertTokenizer,
                          get_linear_schedule_with_warmup, )

RANDOM_SEED = 42
MAX_TOKEN_COUNT = 512  # DistilBERT's maximum token length
STRIDE = 256  # Stride for the sliding window (replaced CHUNK_OVERLAP)
N_EPOCHS = 10
BATCH_SIZE = 16  # Reduced batch size to allow for sliding window approach
GRADIENT_ACCUMULATION_STEPS = 4  # Increased to compensate for smaller batches
EVAL_FREQUENCY = 2  # Reduced validation frequency
HIDDEN_DROPOUT_PROB = 0.3
ATTENTION_PROBS_DROPOUT_PROB = 0.3
BERT_MODEL_NAME = "distilbert-base-uncased"
DB_PATH = "/teamspace/studios/this_studio/echr_cases_anonymized.sqlite"

model_name = "echr_judgments_classifier"
model_dir = "/teamspace/studios/this_studio/echr_distilbert_model_sliding_window"
LR_CONFIG = model_dir + "/lr_config.json"

pl.seed_everything(RANDOM_SEED)
nltk.download("stopwords")
nltk.download("wordnet")
torch.set_float32_matmul_precision("medium")
os.makedirs(model_dir, exist_ok=True)


class JudgmentsDataset(Dataset):
    def __init__(self, data: pd.DataFrame, tokenizer: DistilBertTokenizer, max_token_len: int = MAX_TOKEN_COUNT,
            stride: int = STRIDE):
        self.tokenizer = tokenizer
        self.data = data[['judgment'] + ARTICLES_COLUMNS].copy()
        self.max_token_len = max_token_len
        self.stride = stride

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        judgment = data_row.judgment
        labels = data_row[ARTICLES_COLUMNS]
        labels = labels.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Tokenize the text
        tokenized = self.tokenizer(judgment, truncation=True, max_length=self.max_token_len, padding="max_length",
                return_overflowing_tokens=True, stride=self.stride, return_tensors="pt")

        # Get number of chunks created by sliding window
        num_chunks = len(tokenized["input_ids"])

        # Create labels tensor (same labels for all chunks)
        labels_tensor = torch.tensor(labels.values.astype(np.float32), dtype=torch.float32)

        return {
                "input_ids": tokenized["input_ids"], "attention_mask": tokenized["attention_mask"],
                "labels": labels_tensor, "num_chunks": num_chunks
                }


class JudgmentsDataModule(pl.LightningDataModule):
    def __init__(self, train_df, test_df, tokenizer, batch_size=16, max_token_len=MAX_TOKEN_COUNT, stride=STRIDE):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.stride = stride

    def setup(self, stage=None):
        self.train_dataset = JudgmentsDataset(self.train_df, self.tokenizer, self.max_token_len, self.stride)

        self.test_dataset = JudgmentsDataset(self.test_df, self.tokenizer, self.max_token_len, self.stride)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, pin_memory=True,
                collate_fn=self.sliding_window_collate_fn, )

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2,
                collate_fn=self.sliding_window_collate_fn, )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=2,
                collate_fn=self.sliding_window_collate_fn, )

    def sliding_window_collate_fn(self, batch):
        """
        Custom collate function for sliding window approach
        Processes batches with variable numbers of chunks
        """
        # Extract all chunks and their corresponding labels
        all_input_ids = []
        all_attention_masks = []
        all_labels = []
        batch_boundaries = [0]  # To keep track of which chunks belong to which document

        for item in batch:
            num_chunks = item["num_chunks"]
            all_input_ids.append(item["input_ids"])
            all_attention_masks.append(item["attention_mask"])
            # Repeat labels for each chunk
            all_labels.extend([item["labels"]] * num_chunks)
            batch_boundaries.append(batch_boundaries[-1] + num_chunks)

        # Concatenate all chunks into single tensors
        input_ids = torch.cat(all_input_ids, dim=0)
        attention_mask = torch.cat(all_attention_masks, dim=0)
        labels = torch.stack(all_labels)

        return {
                "input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
                "batch_boundaries": batch_boundaries
                }


class JudgmentsTagger(pl.LightningModule):
    def __init__(self, n_classes: int, batch_size=16, learning_rate=0.001, class_weights=None):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.bert = DistilBertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True, )

        # Enable gradient checkpointing for memory efficiency
        self.bert.gradient_checkpointing_enable()

        # Simplified classifier
        self.classifier = nn.Sequential(nn.Dropout(ATTENTION_PROBS_DROPOUT_PROB),
                nn.Linear(self.bert.config.hidden_size, self.bert.config.hidden_size // 2), nn.GELU(),
                nn.Dropout(HIDDEN_DROPOUT_PROB), nn.Linear(self.bert.config.hidden_size // 2, n_classes))

        self.n_training_steps = None
        self.n_warmup_steps = None
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)

    def forward(self, input_ids, attention_mask, batch_boundaries=None):
        # Process all chunks at once
        def bert_forward():
            return self.bert(input_ids, attention_mask=attention_mask)

        with torch.cuda.amp.autocast():
            bert_output = checkpoint(bert_forward)
            # Use [CLS] token as representation
            pooled_output = bert_output.last_hidden_state[:, 0]

            # Get predictions for all chunks
            chunk_logits = self.classifier(pooled_output)

            # Aggregate predictions for each document using batch_boundaries
            if batch_boundaries is not None:
                # Initialize final outputs tensor
                batch_size = len(batch_boundaries) - 1
                final_outputs = torch.zeros(batch_size, chunk_logits.size(-1), device=chunk_logits.device)

                # For each document, aggregate its chunks
                for i in range(batch_size):
                    start_idx = batch_boundaries[i]
                    end_idx = batch_boundaries[i + 1]

                    # Take max across chunks for each class
                    doc_chunks = chunk_logits[start_idx:end_idx]
                    final_outputs[i] = torch.max(doc_chunks, dim=0)[0]

                return final_outputs, chunk_logits

            return chunk_logits, None

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        batch_boundaries = batch["batch_boundaries"]

        doc_outputs, chunk_outputs = self(input_ids, attention_mask, batch_boundaries)

        # Calculate loss on document-level predictions
        loss = self.criterion(doc_outputs, labels)

        self.log("train_loss", loss, prog_bar=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        batch_boundaries = batch["batch_boundaries"]

        doc_outputs, chunk_outputs = self(input_ids, attention_mask, batch_boundaries)

        # Calculate loss on document-level predictions
        loss = self.criterion(doc_outputs, labels)

        self.log("val_loss", loss, prog_bar=True, logger=True)

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        batch_boundaries = batch["batch_boundaries"]

        doc_outputs, chunk_outputs = self(input_ids, attention_mask, batch_boundaries)

        # Calculate metrics
        preds = torch.sigmoid(doc_outputs)
        preds_binary = (preds > 0.5).float()

        # Calculate accuracy for each class
        accuracy = (preds_binary == labels).float().mean(dim=0)

        # Log average accuracy
        self.log("test_accuracy", accuracy.mean(), logger=True)

        return {"test_preds": preds, "test_labels": labels}

    def configure_optimizers(self):
        # Use AdamW optimizer
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, weight_decay=0.01)

        if self.trainer and self.trainer.datamodule:
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            self.n_training_steps = steps_per_epoch * self.trainer.max_epochs // GRADIENT_ACCUMULATION_STEPS
            self.n_warmup_steps = int(self.n_training_steps * 0.10)
        else:
            raise ValueError("Trainer or datamodule not initialized")

        # Use linear schedule with warmup
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.n_warmup_steps,
                num_training_steps=self.n_training_steps)

        return {
                "optimizer": optimizer, "lr_scheduler": {
                        "scheduler": scheduler, "interval": "step", "frequency": 1, "monitor": "val_loss",
                        }
                }


def calculate_multilabel_class_weights(y: np.ndarray):
    num_labels = y.shape[1]
    class_weights = []

    for i in range(num_labels):
        label_targets = y[:, i]
        unique_classes = np.unique(label_targets)

        if len(unique_classes) == 1:
            class_weights.append(1.0)
        else:
            weights = compute_class_weight(class_weight="balanced", classes=unique_classes, y=label_targets)
            # Weight for the positive class (label=1)
            class_weights.append(weights[1] if 1 in unique_classes else 1.0)

    return np.array(class_weights)


def fetch_cases():
    try:
        print("Connecting to DB...")
        sqlite_connection = sqlite3.connect(DB_PATH)
        cursor = sqlite_connection.cursor()
        print("DB connection successful")

        print("Fetch cases...")
        query_cases = "select case_id, anonymized_judgement from cases;"
        cursor.execute(query_cases)
        result_cases = cursor.fetchall()

        print("Fetch articles...")
        query_articles = "select normalized_article, case_id from articles where article != '';"
        cursor.execute(query_articles)
        result_articles = cursor.fetchall()

        cursor.close()

        # Create Dataframes
        df_cases = pd.DataFrame(result_cases, columns=["case_id", "judgment"])
        df_articles = pd.DataFrame(result_articles, columns=["article", "case_id"])

        return [df_cases, df_articles]

    except sqlite3.Error as error:
        print("Error occurred - ", error)
        exit()

    finally:
        if sqlite_connection:
            sqlite_connection.close()
            print("SQLite Connection closed")


def save_learning_rate(lr_value, save_path=LR_CONFIG):
    """Save the optimal learning rate to a JSON file."""
    config = {"learning_rate": lr_value}
    with open(save_path, "w") as f:
        json.dump(config, f)
    print(f"Saved learning rate {lr_value} to {save_path}")


def load_learning_rate(load_path=LR_CONFIG, default_lr=5e-5):
    """Load the learning rate from a JSON file if it exists."""
    if os.path.exists(load_path):
        with open(load_path, "r") as f:
            config = json.load(f)
            lr = config.get("learning_rate", default_lr)
        print(f"Loaded learning rate {lr} from {load_path}")
        return lr
    print(f"No saved learning rate found at {load_path}")
    return None


def cleanup():
    """Clean up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    df_cases, df_articles = fetch_cases()

    print("Cases shape: ", df_cases.shape)
    print("Articles shape: ", df_articles.shape)

    # Preprocess and clean up data
    print("Preprocessing data...")
    articles = df_articles.article.unique().tolist()

    # Match cases to articles
    merged_df = df_cases.merge(df_articles, left_on="case_id", right_on="case_id", how="left")
    pivot_df = pd.crosstab(merged_df["case_id"], merged_df["article"])
    cases_to_articles_df = df_cases.merge(pivot_df, left_on="case_id", right_index=True, how="left").fillna(0)

    # Get article columns
    ARTICLES_COLUMNS = cases_to_articles_df.columns.tolist()[2:]
    print("Unique articles in the selected cases: ", len(ARTICLES_COLUMNS))

    # Calculate the number of articles linked to each case
    cases_to_articles_df["num_articles"] = cases_to_articles_df[ARTICLES_COLUMNS].sum(axis=1)

    # Remove cases with no linked articles
    cases_to_articles_df = cases_to_articles_df[cases_to_articles_df.num_articles > 0]
    cases_to_articles_df.drop("num_articles", axis=1, inplace=True)

    print("Processed cases shape:", cases_to_articles_df.shape)

    # Split data
    train_df, val_df = train_test_split(cases_to_articles_df, test_size=0.2, random_state=RANDOM_SEED)
    print("Training shape: ", train_df.shape)
    print("Validation shape: ", val_df.shape)

    # Calculate class weights
    train_labels = train_df[ARTICLES_COLUMNS].values
    class_weights = calculate_multilabel_class_weights(train_labels)

    # Convert weights to a PyTorch tensor
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    # Initialize tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained(BERT_MODEL_NAME)
    tokenizer.save_pretrained(model_dir)

    # Create data module
    data_module = JudgmentsDataModule(train_df, val_df, tokenizer, batch_size=BATCH_SIZE, max_token_len=MAX_TOKEN_COUNT,
            stride=STRIDE)

    # Initialize model
    model = JudgmentsTagger(n_classes=len(ARTICLES_COLUMNS), batch_size=BATCH_SIZE, class_weights=class_weights_tensor)

    # Set up callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=model_dir, filename=model_name, save_top_k=1, monitor="val_loss",
            mode="min", )

    early_stopping_callback = EarlyStopping(monitor="val_loss", patience=3, min_delta=0.001)

    logger = TensorBoardLogger("lightning_logs", name="judgments")

    # Check for saved learning rate or use default
    saved_lr = load_learning_rate(default_lr=5e-5)
    if saved_lr is not None:
        model.learning_rate = saved_lr

    # Create trainer
    trainer = pl.Trainer(accelerator="gpu", devices=1, precision="16-mixed",
            accumulate_grad_batches=GRADIENT_ACCUMULATION_STEPS, max_epochs=N_EPOCHS, strategy="deepspeed_stage_2",
            check_val_every_n_epoch=EVAL_FREQUENCY, gradient_clip_val=1.0, logger=logger,
            callbacks=[early_stopping_callback, checkpoint_callback, DeviceStatsMonitor()], log_every_n_steps=100)

    # Clean up before training
    cleanup()

    # Train the model
    print(f"Training with learning rate: {model.learning_rate}")
    trainer.fit(model=model, datamodule=data_module)

    # Evaluate the model
    trainer.validate(model=model, datamodule=data_module)
    trainer.test(ckpt_path="best", datamodule=data_module)

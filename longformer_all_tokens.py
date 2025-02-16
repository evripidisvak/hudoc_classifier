import os
import re
import sqlite3
import nltk
import json
import gc

print("Defining parameters...")
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:512"
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
    BertModel,
    BertTokenizerFast as BertTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import (
    LongformerConfig,
    LongformerModel,
    LongformerTokenizerFast as LongformerTokenizer,
    DataCollatorWithPadding,
)


RANDOM_SEED = 42
MAX_TOKEN_COUNT = 3072
N_EPOCHS = 10
N_ACCUMULATE_BATCHES = 2
BATCH_SIZE = 16
GRADIENT_ACCUMULATION_STEPS = 2
EVAL_FREQUENCY = 1
# EVAL_FREQUENCY = 4
SLIDING_WINDOW_STRIDE = 0.8
HIDDEN_DROPOUT_PROB = 0.3
ATTENTION_PROBS_DROPOUT_PROB = 0.3
ATTENTION_WINDOW = 64
BERT_MODEL_NAME = "allenai/longformer-base-4096"
DB_PATH = "/teamspace/studios/this_studio/echr_cases_anonymized.sqlite"

model_name = "echr_judgments_classifier"
model_dir = "/teamspace/studios/this_studio/echr_longformer_model"
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
        tokenizer: LongformerTokenizer,
        max_token_len: int = 2048,
        sliding_window_stride: float = SLIDING_WINDOW_STRIDE,
    ):
        self.tokenizer = tokenizer
        # self.data = data
        self.data = data[['judgment'] + ARTICLES_COLUMNS].copy()
        self.max_token_len = max_token_len
        self.stride = int(max_token_len * sliding_window_stride)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        data_row = self.data.iloc[index]
        judgment = data_row.judgment
        labels = data_row[ARTICLES_COLUMNS]
        # Convert labels to float32
        labels = labels.apply(pd.to_numeric, errors="coerce").fillna(0)

        # Use sliding window tokenization
        encodings = self.tokenizer(
            judgment,
            max_length=self.max_token_len,
            stride=self.stride,
            return_overflowing_tokens=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Handle multiple chunks
        num_chunks = len(encodings["input_ids"])
        print(f"Judgment split into {num_chunks} chunks")

        # Create labels tensor for all chunks
        labels_tensor = torch.tensor(
            labels.values.astype(np.float32), dtype=torch.float32
        )

        # Create individual samples for each chunk
        chunked_samples = []
        for i in range(num_chunks):
            chunked_samples.append(
                {
                    "input_ids": encodings["input_ids"][i],
                    "attention_mask": encodings["attention_mask"][i],
                    "labels": labels_tensor,
                }
            )

        return chunked_samples  # Returns a list of samples


class JudgmentsDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        test_df,
        tokenizer,
        batch_size=8,
        max_token_len=2048,
        sliding_window_stride= SLIDING_WINDOW_STRIDE,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.train_df = train_df
        self.test_df = test_df
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len
        self.sliding_window_stride = sliding_window_stride

        self.data_collator = DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding="max_length",
            max_length=self.max_token_len,
            return_tensors="pt",
        )

    def setup(self, stage=None):
        self.train_dataset = JudgmentsDataset(
            self.train_df,
            self.tokenizer,
            self.max_token_len,
            sliding_window_stride=self.sliding_window_stride,
        )

        self.test_dataset = JudgmentsDataset(
            self.test_df,
            self.tokenizer,
            self.max_token_len,
            sliding_window_stride=self.sliding_window_stride,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            num_workers=8,
            pin_memory=True,
            prefetch_factor=4,
            persistent_workers=True,
        )

    def collate_fn(self, batch):
        flattened_batch = []
        
        for sample in batch:
            if isinstance(sample, list):  # Ensure each sample is a list of chunks
                flattened_batch.extend(sample)  # Flatten the chunks into one list
            else:
                flattened_batch.append(sample)  # Handle single samples

        # Check if we have an empty batch (this can sometimes happen)
        if len(flattened_batch) == 0:
            raise ValueError("Collate function received an empty batch!")

        # Convert list of dictionaries into a dictionary of lists (needed for tokenizer.pad)
        batch_dict = {key: [dic[key] for dic in flattened_batch] for key in flattened_batch[0].keys()}

        # Debugging: Check the format of batch_dict
        for key, value in batch_dict.items():
            if not isinstance(value, list):
                raise TypeError(f"Expected list for key '{key}', but got {type(value)}")

        # Ensure correct padding using DataCollator
        return self.data_collator(batch_dict)

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
        self, n_classes: int, batch_size=8, learning_rate=0.001, class_weights=None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        config = LongformerConfig.from_pretrained(
            BERT_MODEL_NAME,
            return_dict=True,
            gradient_checkpointing=True,
            attention_window=[ATTENTION_WINDOW] * 12,
            hidden_dropout_prob=HIDDEN_DROPOUT_PROB,
            attention_probs_dropout_prob=ATTENTION_PROBS_DROPOUT_PROB
        )
        self.bert = LongformerModel.from_pretrained(
            BERT_MODEL_NAME,
            config=config,
        )

        self.bert.gradient_checkpointing_enable()

        self.dropout = nn.Dropout(ATTENTION_PROBS_DROPOUT_PROB)
        # self.classifier = nn.Linear(self.bert.config.hidden_size, n_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(ATTENTION_PROBS_DROPOUT_PROB),
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.GELU(),
            nn.Dropout(ATTENTION_PROBS_DROPOUT_PROB),
            nn.Linear(config.hidden_size // 2, n_classes)
        )


        self.n_training_steps = None
        self.n_warmup_steps = None

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        # self.criterion = nn.MSELoss()
    
    def forward(self, input_ids, attention_mask, labels=None):
        def bert_forward():
            return self.bert(input_ids, attention_mask=attention_mask)
            
        with torch.cuda.amp.autocast():
            # Use checkpointing for bert forward pass
            bert_output = checkpoint(bert_forward)
            pooled_output = bert_output.pooler_output
            
            # Process in chunks if needed
            if pooled_output.shape[0] > 1:
                chunk_size = 1
                outputs = []
                for i in range(0, pooled_output.shape[0], chunk_size):
                    chunk = pooled_output[i:i + chunk_size]
                    chunk_output = self.classifier(chunk)
                    outputs.append(chunk_output)
                    del chunk
                    torch.cuda.empty_cache()
                logits = torch.cat(outputs, dim=0)
            else:
                logits = self.classifier(pooled_output)

        output = torch.sigmoid(logits)
        
        loss = 0
        if labels is not None:
            loss = nn.BCEWithLogitsLoss()(logits, labels.float())
        
        # Clean up
        del bert_output, pooled_output, logits
        torch.cuda.empty_cache()
        
        return loss, output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, prog_bar=True, logger=True)

        del input_ids, attention_mask, labels
        torch.cuda.empty_cache()
        gc.collect()

        # return {"loss": loss, "predictions": outputs, "labels": labels}
        return loss

    def calculate_metrics(self, outputs, labels):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        predicted_labels = (outputs > 0.5).int()

        # Per-label metrics
        per_label_metrics = {}
        for i, article in enumerate(ARTICLES_COLUMNS):
            # Per-label accuracy
            label_accuracy = accuracy(
                preds=outputs[:, i], target=labels[:, i], task="binary"
            ).to(device)

            # Per-label precision
            label_precision = precision(
                preds=outputs[:, i], target=labels[:, i], task="binary"
            ).to(device)

            # Per-label recall
            label_recall = recall(
                preds=outputs[:, i], target=labels[:, i], task="binary"
            ).to(device)

            # Per-label F1 score
            label_f1 = f1_score(
                preds=outputs[:, i], target=labels[:, i], task="binary"
            ).to(device)

            # Per-label AUROC
            label_auroc = auroc(
                preds=outputs[:, i],
                target=labels[:, i].type(torch.LongTensor).to(device),
                task="binary",
            ).to(device)

            per_label_metrics[article] = {
                "accuracy": label_accuracy.item(),
                "precision": label_precision.item(),
                "recall": label_recall.item(),
                "f1_score": label_f1.item(),
                "auroc": label_auroc.item(),
            }

            # Log per-label metrics
            self.log(f"{article}_accuracy", label_accuracy, prog_bar=False, logger=True)
            self.log(
                f"{article}_precision", label_precision, prog_bar=False, logger=True
            )
            self.log(f"{article}_recall", label_recall, prog_bar=False, logger=True)
            self.log(f"{article}_f1_score", label_f1, prog_bar=False, logger=True)
            self.log(f"{article}_auroc", label_auroc, prog_bar=False, logger=True)

        # Overall metrics (weighted average)
        overall_accuracy = accuracy(
            preds=outputs,
            target=labels,
            task="multilabel",
            num_labels=len(ARTICLES_COLUMNS),
            average="weighted",
        ).to(device)

        overall_f1_score = f1_score(
            preds=outputs,
            target=labels,
            task="multilabel",
            num_labels=len(ARTICLES_COLUMNS),
            average="weighted",
        ).to(device)

        overall_precision = precision(
            preds=outputs,
            target=labels,
            task="multilabel",
            num_labels=len(ARTICLES_COLUMNS),
            average="weighted",
        ).to(device)

        overall_recall = recall(
            preds=outputs,
            target=labels,
            task="multilabel",
            num_labels=len(ARTICLES_COLUMNS),
            average="weighted",
        ).to(device)

        overall_auroc = auroc(
            preds=outputs,
            target=labels.type(torch.LongTensor).to(device),
            task="multilabel",
            num_labels=len(ARTICLES_COLUMNS),
            average="weighted",
        ).to(device)

        # Log overall metrics
        self.log("overall_accuracy", overall_accuracy, prog_bar=True, logger=True)
        self.log("overall_f1_score", overall_f1_score, prog_bar=True, logger=True)
        self.log("overall_precision", overall_precision, prog_bar=True, logger=True)
        self.log("overall_recall", overall_recall, prog_bar=True, logger=True)
        self.log("overall_auroc", overall_auroc, prog_bar=True, logger=True)

        # Debugging outputs
        print("\nPer-Label Metrics:")
        for article, metrics in per_label_metrics.items():
            print(f"{article}:")
            for metric, value in metrics.items():
                print(f"  {metric}: {value}")

        print("\nOverall Metrics:")
        print(f"Accuracy: {overall_accuracy}")
        print(f"F1 Score: {overall_f1_score}")
        print(f"Precision: {overall_precision}")
        print(f"Recall: {overall_recall}")
        print(f"AUROC: {overall_auroc}")

        return {
            "per_label_metrics": per_label_metrics,
            "overall_metrics": {
                "accuracy_score": overall_accuracy,
                "f1_score": overall_f1_score,
                "precision_score": overall_precision,
                "recall_score": overall_recall,
                "multilabel_auroc": overall_auroc,
            },
        }

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("val_loss", loss, prog_bar=True, logger=True)

        del input_ids, attention_mask, labels
        torch.cuda.empty_cache()
        gc.collect()

        return loss

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("test_loss", loss, prog_bar=True, logger=True)

        metrics = self.calculate_metrics(outputs, labels)

        del input_ids, attention_mask, labels
        torch.cuda.empty_cache()
        gc.collect()

        return {"test_loss": loss, "metrics": metrics}

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate)
        # optimizer = deepspeed.ops.adam.DeepSpeedCPUAdam(self.parameters(), lr=self.learning_rate)

        # Dynamically calculate training steps and warmup steps
        if self.trainer and self.trainer.datamodule:
            steps_per_epoch = len(self.trainer.datamodule.train_dataloader())
            self.n_training_steps = steps_per_epoch * self.trainer.max_epochs
            self.n_warmup_steps = int(self.n_training_steps * 0.10)
        else:
            raise ValueError("Trainer or datamodule not initialized. Cannot compute training steps.")

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.n_warmup_steps,
            num_training_steps=self.n_training_steps
        )

        # Add this configuration to ensure proper ordering
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
        query_articles = "select article, case_id from articles where article != '';"
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

    config = LongformerConfig.from_pretrained(
        BERT_MODEL_NAME,
        return_dict=True,
        gradient_checkpointing=True,
        attention_window=[ATTENTION_WINDOW] * 12,
        hidden_dropout_prob=HIDDEN_DROPOUT_PROB,
        attention_probs_dropout_prob=ATTENTION_PROBS_DROPOUT_PROB
    )
    # config.attention_window = [128] * config.num_hidden_layers  # Reduce from 512 to 128
    bert_model = LongformerModel.from_pretrained(
        BERT_MODEL_NAME,
        config=config,
    )
    # Enable Memory Optimizations
    bert_model = torch.compile(bert_model)  # Speed up training
    bert_model = bert_model.to(memory_format=torch.channels_last)  # Optimize tensor memory layout

    # bert_model = LongformerModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
    tokenizer = LongformerTokenizer.from_pretrained(BERT_MODEL_NAME)

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
    tuner = Tuner(trainer)

    # First check if we have a saved learning rate
    saved_lr = load_learning_rate()

    if saved_lr is None:
        print("Finding optimal learning rate...")
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

    # print("scale_batch_size")
    # tuner.scale_batch_size(
    #     model=model, mode="binsearch", datamodule=data_module, init_val=5
    # )
    # print("lr_finder")
    # lr_finder = tuner.lr_find(model=model, datamodule=data_module)
    # print(lr_finder.results)

    print(f"Training with learning rate: {model.learning_rate}")

    print(BERT_MODEL_NAME)
    print("fit")
    cleanup()
    trainer.fit(model=model, datamodule=data_module)

    print("vaidate")
    trainer.validate(model=model, datamodule=data_module)
    print("test")
    trainer.test(ckpt_path="best", datamodule=data_module)

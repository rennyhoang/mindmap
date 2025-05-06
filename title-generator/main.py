#!/usr/bin/env python3
"""
Fine-tune BART on Newsroom to generate article titles.

Requires: transformers, datasets, torch, sentencepiece, evaluate, rouge_score
"""

import logging
import torch
from datasets import load_dataset, load_from_disk, DatasetDict
from evaluate import load as load_metric
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import os

# Hyperparameters
MAX_INPUT_LENGTH = 1024
MAX_TARGET_LENGTH = 64
BATCH_SIZE = 1
LEARNING_RATE = 3e-5
NUM_EPOCHS = 1
MODEL_NAME = "facebook/bart-large"
TOKENIZED_PATH = "tokenized_newsroom"

def preprocess_fn(examples):
    """Tokenize inputs (articles) and targets (titles)."""
    inputs = tokenizer(
        examples["text"],
        max_length=MAX_INPUT_LENGTH,
        truncation=True,
        padding="max_length",
    )
    targets = tokenizer(
        examples["title"],
        max_length=MAX_TARGET_LENGTH,
        truncation=True,
        padding="max_length",
    )
    inputs["labels"] = targets["input_ids"]
    return inputs


def compute_metrics(pred):
    """Compute ROUGE scores for generated titles vs. references."""
    labels = pred.label_ids
    preds = pred.predictions
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # remove -100 from labels
    labels = [[l for l in lab if l != -100] for lab in labels]
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    result = rouge.compute(
        predictions=decoded_preds,
        references=decoded_labels,
        use_stemmer=True,
    )
    return {k: v.mid.fmeasure * 100 for k, v in result.items()}


def generate_title(article: str) -> str:
    """Generate a title for a single article string."""
    inputs = tokenizer(
        article,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_INPUT_LENGTH,
    ).to(model.device)
    out = model.generate(
        **inputs,
        max_length=MAX_TARGET_LENGTH,
        num_beams=5,
        early_stopping=True,
    )
    return tokenizer.decode(out[0], skip_special_tokens=True)


def main():
    logging.basicConfig(level=logging.INFO)

    # 1. Load dataset
    ds = load_dataset("newsroom", "all", data_dir="./newsroom-release/release")
    train_ds = ds["train"]
    val_ds = ds["validation"]

    # 2. Load tokenizer & model
    global tokenizer, model, rouge
    tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
    model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)
    rouge = load_metric("rouge")

    if os.path.isdir(TOKENIZED_PATH):
        logging.info(f"Loading tokenized from {TOKENIZED_PATH}/")
        tok_ds = load_from_disk(TOKENIZED_PATH)
        train_tok = tok_ds["train"]
        val_tok = tok_ds["validation"]
    else:
        logging.info("Tokenizing for the first time…")

        train_tok = train_ds.map(
            preprocess_fn,
            batched=True,
            remove_columns=train_ds.column_names,
        )
        val_tok = val_ds.map(
            preprocess_fn,
            batched=True,
            remove_columns=val_ds.column_names,
        )
        
        tok_ds = DatasetDict({"train": train_tok, "validation": val_tok})
        tok_ds.save_to_disk(TOKENIZED_PATH)

    # 4. Data collator & training args
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    training_args = Seq2SeqTrainingArguments(
        output_dir="bart-newsroom-title",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        num_train_epochs=NUM_EPOCHS,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
    )

    # 5. Initialize trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # 6. Train
    trainer.train()

    # 7. Generate a sample title
    sample = val_ds[0]["article"]
    print("Predicted title:", generate_title(sample))
    print("Reference   :", val_ds[0]["title"])


if __name__ == "__main__":
    main()
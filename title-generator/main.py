import torch
from transformers import BartForConditionalGeneration, BartTokenizer
from datasets import load_dataset
import os
from datetime import datetime
import json


def generate_title(text, model, tokenizer, device, max_length=15):
    # Prepare the text input
    inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True).to(
        device
    )

    # Generate summary
    summary_ids = model.generate(
        inputs["input_ids"],
        num_beams=4,
        min_length=3,
        max_length=max_length,
        early_stopping=True,
        no_repeat_ngram_size=2,
        repetition_penalty=1.5,
        length_penalty=1.0,
    )

    # Decode the generated tokens
    title = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    # Post-process the title
    title = title.strip()
    title = " ".join(title.split())  # Remove extra spaces

    # Capitalize first letter
    if title:
        title = title[0].upper() + title[1:] if len(title) > 1 else title.upper()

    return title


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load pre-trained model and tokenizer
    print("Loading pre-trained BART model...")
    model_name = "facebook/bart-large-cnn"  # Pre-trained on CNN/DM for summarization
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

    # Test with sample texts
    sample_texts = [
        "The researchers discovered a new species of butterfly in the Amazon rainforest. The discovery was made during an expedition led by Dr. Maria Santos, who has been studying Amazonian butterflies for over a decade. The new species has distinctive blue and gold patterns on its wings.",
        "A breakthrough in renewable energy technology was announced today. Scientists at MIT have developed a new type of solar panel that can generate electricity even on cloudy days. The technology uses a special coating that captures a wider spectrum of light.",
        "The local community came together to build a new playground. Over 100 volunteers worked for three weekends to construct the playground, which includes swings, slides, and climbing equipment. The project was funded by donations from local businesses.",
    ]

    print("\nGenerating titles for sample texts:")
    for i, text in enumerate(sample_texts, 1):
        title = generate_title(text, model, tokenizer, device)
        print(f"\nSample {i}:")
        print(f"Text: {text[:100]}...")
        print(f"Generated title: {title}")

    # Optional: Fine-tune on your specific dataset
    print("\nWould you like to fine-tune the model on CNN/DM dataset? (y/n)")
    choice = input()
    if choice.lower() == "y":
        fine_tune_model(model, tokenizer, device)


def fine_tune_model(model, tokenizer, device):
    from transformers import Trainer, TrainingArguments
    from torch.utils.data import Dataset

    class SummarizationDataset(Dataset):
        def __init__(
            self, texts, summaries, tokenizer, max_length=1024, summary_max_length=64
        ):
            self.tokenizer = tokenizer
            self.inputs = []
            self.targets = []

            for text, summary in zip(texts, summaries):
                # Tokenize inputs
                inputs = tokenizer(
                    text, max_length=max_length, padding="max_length", truncation=True
                )

                # Tokenize targets
                with tokenizer.as_target_tokenizer():
                    targets = tokenizer(
                        summary,
                        max_length=summary_max_length,
                        padding="max_length",
                        truncation=True,
                    )

                self.inputs.append(inputs)
                self.targets.append(targets)

        def __len__(self):
            return len(self.inputs)

        def __getitem__(self, idx):
            input_ids = torch.tensor(self.inputs[idx]["input_ids"])
            attention_mask = torch.tensor(self.inputs[idx]["attention_mask"])

            target_ids = torch.tensor(self.targets[idx]["input_ids"])
            target_attention_mask = torch.tensor(self.targets[idx]["attention_mask"])

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": target_ids,
                "decoder_attention_mask": target_attention_mask,
            }

    # Load dataset
    print("Loading CNN/DM dataset...")
    dataset = load_dataset("cnn_dailymail", "3.0.0")

    # Create training dataset (using a small subset for quick fine-tuning)
    train_size = 1000  # Adjust as needed
    train_texts = dataset["train"]["article"][:train_size]
    train_summaries = dataset["train"]["highlights"][:train_size]

    train_dataset = SummarizationDataset(train_texts, train_summaries, tokenizer)

    # Set up training arguments
    training_args = TrainingArguments(
        output_dir="./bart_title_generator",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=3e-5,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
    )

    # Train the model
    print("Fine-tuning the model...")
    trainer.train()

    # Save the fine-tuned model
    model_path = "./bart_title_generator/fine_tuned_model"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"Fine-tuned model saved to {model_path}")

    # Test the fine-tuned model
    sample_texts = [
        "The researchers discovered a new species of butterfly in the Amazon rainforest.",
        "A breakthrough in renewable energy technology was announced today.",
        "The local community came together to build a new playground.",
    ]

    print("\nGenerating titles with fine-tuned model:")
    for i, text in enumerate(sample_texts, 1):
        title = generate_title(text, model, tokenizer, device)
        print(f"\nSample {i}:")
        print(f"Text: {text}")
        print(f"Generated title: {title}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nProcess interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print("\nProgram finished")

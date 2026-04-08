"""Run a GPU training job on Hugging Face infrastructure."""

from jobs_decorator import job


@job(
    flavor="a10g-small",
    timeout="2h",
    dependencies=["torch", "transformers", "datasets", "accelerate"],
)
def train(model_name: str, learning_rate: float = 5e-5, epochs: int = 3):
    from datasets import load_dataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
    dataset = load_dataset("imdb", split="train[:1000]")

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, max_length=512)

    dataset = dataset.map(tokenize, batched=True)

    args = TrainingArguments(
        output_dir="/tmp/output",
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=8,
    )

    trainer = Trainer(model=model, args=args, train_dataset=dataset)
    trainer.train()

    return {"final_loss": trainer.state.log_history[-1].get("loss")}


handle = train.remote("bert-base-uncased", learning_rate=3e-5)
print(f"Training job: {handle.url}")

# Check status while it runs
print(f"Status: {handle.status()}")

# Stream logs
for line in handle.logs():
    print(line)

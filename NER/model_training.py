# Importing necessary libraries
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer


# This function converts text to token-level input for training our model
def encode_with_labels(data, tokenizer):
    all_inputs = tokenizer(data["text"], truncation=True, padding="max_length", max_length=128, return_offsets_mapping=True) # Tokenize all our sentences

    # Creating label for each token. 0 - "O" (no entity), 1 - "B-MOUNTAIN" (begining of mountain name), 2 - "I-MOUNTAIN" (inside of mountain name)
    all_labels = []
    for i, entities in enumerate(data["entities"]):
        labels = [0] * len(all_inputs["input_ids"][i])
        for ent in entities:
            start, end, label = ent["start"], ent["end"], ent["label"]
            for idx, (token_start, token_end) in enumerate(all_inputs["offset_mapping"][i]):
                if token_start is None or token_end is None:
                    continue
                if token_start >= start and token_end <= end:
                    labels[idx] = 1 if token_start == start else 2
        all_labels.append(labels)

    # Attaching labels to our tokens
    all_inputs["labels"] = all_labels
    return all_inputs


if __name__ == "__main__":
    # Loading dataset
    train_dataset = load_dataset("json", data_files="train.jsonl")["train"]
    val_dataset = load_dataset("json", data_files="validation.jsonl")["train"]

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased")
    train_dataset = train_dataset.map(lambda x: encode_with_labels(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: encode_with_labels(x, tokenizer), batched=True)

    # Preparing model
    label2id = {"O": 0, "B-MOUNTAIN": 1, "I-MOUNTAIN": 2}
    id2label = {v: k for k, v in label2id.items()}
    
    # Getting model that we will fine-tune. Setting its output labels
    model = AutoModelForTokenClassification.from_pretrained(
        "distilbert-base-cased",
        num_labels=3,
        id2label=id2label,
        label2id=label2id,
    )

    # Setting arguments for training
    args = TrainingArguments(
        output_dir="./mountain-ner",
        learning_rate=3e-5, # Important to take small learning rate so already trained model will carefully adjust its coefficient for our task
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4, 
        num_train_epochs=3, # 3 epochs of training
        weight_decay=0.01,
        fp16=True, # Training with GPU so it will be mush faster
        logging_steps=200, # Information about training will be printed every 200 steps
        save_strategy="epoch",
        eval_strategy="epoch",
    )

    # Giving to Trainer our pretrained model, training args, datasets and tokenizer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
    )

    # Training model
    trainer.train()

    # Evaluating result and then saving model so we don't need to train it again
    trainer.evaluate()
    trainer.save_model("./mountain-ner-distilbert")
    tokenizer.save_pretrained("./mountain-ner-model")

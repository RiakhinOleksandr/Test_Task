# Importing necessary libraries
import random
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from seqeval.metrics import classification_report


# Convert both true and predicted entities to comparable format
def to_entity_list(entities):
    """Convert list of dicts with spans into list of (label, start, end)"""
    return [(e["label"], e["start"], e["end"]) for e in entities]


def to_seqeval_format(entities_batch):
    """Convert for seqeval metric input"""
    # Each text → list of entity labels
    return [[e[0] for e in to_entity_list(ents)] for ents in entities_batch]


if __name__ == "__main__":
    # Loading test dataset
    test_data = load_dataset("json", data_files="test.jsonl")["train"]

    id2label = {0: "O", 1: "B-MOUNTAIN", 2: "I-MOUNTAIN"}
    label2id = {"O": 0, "B-MOUNTAIN": 1, "I-MOUNTAIN": 2}

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained("./mountain-ner-model")
    model = AutoModelForTokenClassification.from_pretrained(
        "./mountain-ner-distilbert",
        id2label=id2label,
        label2id=label2id
    )

    # Creating pipeline for working with our model
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device = 0 # We will use GPU so evaluating will much faster
    )

    texts = [ex["text"] for ex in test_data]
    true_entities = [ex["entities"] for ex in test_data]

    # Getting predictions from our model
    predictions = ner_pipeline(texts, batch_size=16)

    y_true = to_seqeval_format(true_entities)
    y_pred = []
    for preds in predictions:
        y_pred.append([p["entity_group"] for p in preds])

    # Compute metrics
    print(classification_report(y_true, y_pred, digits=5))

    # Looking at number of wrong predictions and on some of them
    wrong = []

    for i, (text, true_ents, pred_ents) in enumerate(zip(texts, true_entities, predictions)):
        # Normalize entities to simple (label, start, end) tuples
        true_set = {(e["label"], e["start"], e["end"]) for e in true_ents}
        pred_set = {(p["entity_group"], p["start"], p["end"]) for p in pred_ents}
    
        if true_set != pred_set: # If actual entities and predictes ones dismatch adding info of them to list
            wrong.append({
                "index": i,
                "text": text,
                "true_entities": list(true_set),
                "pred_entities": list(pred_set)
            })
    
    print(f"Total samples: {len(texts)}")
    print(f"Wrong predictions: {len(wrong)}\n")

    # If there are low amount of mistake looking at all of them. Otherwise looking on randomly chosen ones
    if len(wrong) <= 15:
        for mistake in wrong:
            print(f"Text: {mistake['text']}")
            print(f"True: {mistake['true_entities']}")
            print(f"Pred: {mistake['pred_entities']}")
    else:
        mistakes = random.sample(wrong, 15)
        for mistake in mistakes:
            print(f"Text: {mistake['text']}")
            print(f"True: {mistake['true_entities']}")
            print(f"Pred: {mistake['pred_entities']}")
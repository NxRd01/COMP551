from datasets import load_dataset
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import numpy as np


bert_path = Path("/opt/models/bert-base-uncased")
gpt2_path = Path("/opt/models/distilgpt2")

# Load all splits of the GoEmotions dataset
ds = load_dataset("google-research-datasets/go_emotions", "simplified")

# Function to simplify and process a dataset split
def simplify_split(split):
    df = split.to_pandas()
    df = df[df['labels'].apply(lambda x: len(x) == 1)]
    return df.reset_index(drop=True)

# Process all splits
train_df = simplify_split(ds['train'])
validation_df = simplify_split(ds['validation'])
test_df = simplify_split(ds['test'])

print("Train set shape:", train_df.shape)
print("Validation set shape:", validation_df.shape)
print("Test set shape:", test_df.shape)

def classify_with_model(model_name, test_data):
    # Load model and tokenizer
    model = AutoModel.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Add padding token for GPT-2
    if model_name == gpt2_path:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    # Convert test data to numpy arrays
    test_texts = np.array(test_data['text'].tolist())
    test_labels = np.array(test_data['labels'].tolist())

    # Tokenize data
    test_encodings = tokenizer(test_texts.tolist(), truncation=True, padding=True, return_tensors="pt")

    # Create DataLoader
    test_dataset = TensorDataset(test_encodings['input_ids'], test_encodings['attention_mask'], torch.tensor(test_labels))
    test_loader = DataLoader(test_dataset, batch_size=32)

    # Classification function
    def classify(loader):
        model.eval()
        predictions = []
        true_labels = []
        with torch.no_grad():
            for batch in loader:
                input_ids, attention_mask, labels = batch
                outputs = model(input_ids, attention_mask=attention_mask)
                logits = outputs.last_hidden_state.mean(dim=1)  # Use mean pooling
                preds = torch.argmax(logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        return accuracy_score(true_labels, predictions)

    # Evaluate on test set
    test_accuracy = classify(test_loader)
    print(f"Model: {model_name}, Test Accuracy:{test_accuracy:.4f}")
    return {
        'Model': model_name,
        'Test Accuracy': f'{test_accuracy:.4f}'
    }


# Classify using BERT and GPT-2
results = []

for model_name in [bert_path, gpt2_path]:
    result = classify_with_model(model_name,test_df)
    results.append(result)

# Print results
for result in results:
    print(result)
from datasets import load_dataset
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import random 
import torch.nn as nn

# Define emotion labels
emotion_labels = {
    0: 'admiration', 1: 'amusement', 2: 'anger', 3: 'annoyance', 4: 'approval', 5: 'caring',
    6: 'confusion', 7: 'curiosity', 8: 'desire', 9: 'disappointment', 10: 'disapproval',
    11: 'disgust', 12: 'embarrassment', 13: 'excitement', 14: 'fear', 15: 'gratitude',
    16: 'grief', 17: 'joy', 18: 'love', 19: 'nervousness', 20: 'optimism', 21: 'pride',
    22: 'realization', 23: 'relief', 24: 'remorse', 25: 'sadness', 26: 'surprise', 27: 'neutral'
}

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

class EmotionClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(EmotionClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.last_hidden_state[:, 0]  # Use [CLS] token for classification
        return self.classifier(pooled_output)

def evaluate_model(model_name, test_df, batch_size=32):
    # Load tokenizer and create model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EmotionClassifier(model_name, num_labels=28)

    # Add padding token for GPT-2
    if model_name == gpt2_path:
        tokenizer.pad_token = tokenizer.eos_token
        model.model.config.pad_token_id = model.model.config.eos_token_id

    # Tokenize all texts
    encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, return_tensors="pt")
    
    # Create dataset and dataloader
    dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'], torch.tensor(test_df['labels'].tolist()))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Evaluation
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids, attention_mask, labels = batch
            outputs = model(input_ids, attention_mask)
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # Generate classification report
    report = classification_report(all_labels, all_predictions, target_names=[emotion_labels[i] for i in range(28)], zero_division=0)

    print(f"Model: {model_name}")
    print(f"Accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(report)

    return accuracy, report

def test_model_predictions(model_name, test_texts, true_labels, num_samples=5):
    # Load tokenizer and create model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EmotionClassifier(model_name, num_labels=28)

    # Add padding token for GPT-2
    if model_name == gpt2_path:
        tokenizer.pad_token = tokenizer.eos_token
        model.model.config.pad_token_id = model.model.config.eos_token_id

    # Randomly select samples
    indices = random.sample(range(len(test_texts)), num_samples)
    
    for idx in indices:
        text = test_texts[idx]
        true_label = true_labels[idx]
        
        # Ensure true_label is a scalar
        if isinstance(true_label, np.ndarray):
            true_label = true_label.item()
        
        # Tokenize and process the text
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        
        # Get model prediction
        with torch.no_grad():
            logits = model(**inputs)
            pred = torch.argmax(logits, dim=1).item()
        
        print(f"Model: {model_name}")
        print(f"Text: {text}")
        print(f"True emotion: {emotion_labels[true_label]}")
        print(f"Predicted emotion: {emotion_labels[pred]}")
        print("---")

# # Usage example
# test_model_predictions(bert_path, test_df['text'], test_df['labels'])
# test_model_predictions(gpt2_path, test_df['text'], test_df['labels'])


accuracy_bert, report_bert = evaluate_model(bert_path, test_df)
accuracy_gpt, report_gpt = evaluate_model(gpt2_path, test_df)



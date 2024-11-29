from datasets import load_dataset
import pandas as pd
from pathlib import Path
import torch
from transformers import AutoModel, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import random 
import torch.nn as nn
from tqdm import tqdm


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

#Create the models 
class BertClassifier(nn.Module):
    def __init__(self, bert_model, num_labels=28):
        super().__init__()
        self.bert = bert_model
        self.dropout1 = nn.Dropout(0.3)  # Increased dropout
        self.dense = nn.Linear(768, 512)
        self.dropout2 = nn.Dropout(0.2)
        self.classifier = nn.Linear(512, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs[0][:, 0, :]
        pooled_output = self.dropout1(pooled_output)
        dense_output = torch.relu(self.dense(pooled_output))
        dense_output = self.dropout2(dense_output)
        logits = self.classifier(dense_output)
        return logits

class GPT2Classifier(nn.Module):
    def __init__(self, gpt2_model, num_labels=28):
        super().__init__()
        self.gpt2 = gpt2_model
        self.dropout1 = nn.Dropout(0.3)
        self.dense1 = nn.Linear(768, 512)
        self.batch_norm = nn.BatchNorm1d(512)
        self.dropout2 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(512, 256)
        self.classifier = nn.Linear(256, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs[0]
        
        # Use attention mask for better pooling
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        masked_hidden = hidden_states * mask
        summed = torch.sum(masked_hidden, dim=1)
        count = torch.clamp(mask.sum(1), min=1e-9)
        pooled_output = summed / count
        
        # Enhanced feature extraction
        x = self.dropout1(pooled_output)
        x = self.dense1(x)
        x = self.batch_norm(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = torch.relu(self.dense2(x))
        logits = self.classifier(x)
        return logits
    
    
#Data preparation 
def prepare_data(texts, labels, tokenizer, max_length=128):
    encodings = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    dataset = TensorDataset(
        encodings['input_ids'],
        encodings['attention_mask'],
        torch.tensor(labels)
    )
    
    return dataset

# Prepare datasets for both models
def create_dataloaders(train_df, val_df, test_df, tokenizer, batch_size=32):
    train_dataset = prepare_data(
        train_df['text'].tolist(),
        train_df['labels'].apply(lambda x: x[0]).tolist(),
        tokenizer
    )
    val_dataset = prepare_data(
        val_df['text'].tolist(),
        val_df['labels'].apply(lambda x: x[0]).tolist(),
        tokenizer
    )
    test_dataset = prepare_data(
        test_df['text'].tolist(),
        test_df['labels'].apply(lambda x: x[0]).tolist(),
        tokenizer
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader

#Train model
def train_model(model, train_loader, val_loader, device, num_epochs=3, learning_rate=2e-5):
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    
    best_val_acc = 0
    best_model_state = None
    model = model.to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []
        
        epoch_loss = 0
        num_batches = 0
        
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_preds.extend(preds.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
        
        # Validation
        model.eval()
        val_preds = []
        val_labels = []
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = [b.to(device) for b in batch]
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_batches += 1
                preds = torch.argmax(outputs, dim=1)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
        
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        avg_train_loss = epoch_loss / num_batches
        avg_val_loss = val_loss / val_batches
        
        print(f'\nEpoch {epoch + 1}:')
        print(f'Average Training Loss: {avg_train_loss:.4f}')
        print(f'Average Validation Loss: {avg_val_loss:.4f}')
        print(f'Training Accuracy: {train_acc:.4f}')
        print(f'Validation Accuracy: {val_acc:.4f}')
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f'New best validation accuracy: {val_acc:.4f}')
    
    return best_model_state


def evaluate_test_set(model, test_loader, device):
    model.eval()
    test_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs, dim=1)
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(test_labels, test_preds)
    print("\nTest Set Results:")
    print(f'Test Accuracy: {test_acc:.4f}')
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds))
    
    return test_acc


# Hyperparameters
hyperparameters = {
    'learning_rate': 1e-5,
    'num_epochs': 10,
    'batch_size': 16,
    'max_length': 128,
    'warmup_steps': 100,
    'weight_decay': 0.01
}

# Set device and random seed
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Load tokenizers and models
bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
gpt2_tokenizer = AutoTokenizer.from_pretrained(gpt2_path)
gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token

bert_base = AutoModel.from_pretrained(bert_path)
gpt2_base = AutoModel.from_pretrained(gpt2_path)

# Create classifiers
bert_classifier = BertClassifier(bert_base)
gpt2_classifier = GPT2Classifier(gpt2_base)

# Prepare dataloaders for both models
bert_loaders = create_dataloaders(
    train_df, 
    validation_df, 
    test_df, 
    bert_tokenizer, 
    batch_size=hyperparameters['batch_size']
)
gpt2_loaders = create_dataloaders(
    train_df, 
    validation_df, 
    test_df, 
    gpt2_tokenizer, 
    batch_size=hyperparameters['batch_size']
)

# Train BERT
print("Training BERT classifier...")
bert_best_state = train_model(
    bert_classifier, 
    bert_loaders[0], 
    bert_loaders[1], 
    device,
    num_epochs=hyperparameters['num_epochs'],
    learning_rate=hyperparameters['learning_rate']
)

bert_classifier.load_state_dict(bert_best_state)
evaluate_test_set(bert_classifier, bert_loaders[2], device)

# Train GPT-2
print("\nTraining GPT-2 classifier...")
gpt2_best_state = train_model(
    gpt2_classifier, 
    gpt2_loaders[0], 
    gpt2_loaders[1], 
    device,
    num_epochs=hyperparameters['num_epochs'],
    learning_rate=hyperparameters['learning_rate']
)

gpt2_classifier.load_state_dict(gpt2_best_state)
evaluate_test_set(gpt2_classifier, gpt2_loaders[2], device)

            
            
            
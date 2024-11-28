import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from datasets import load_dataset

emotion_labels = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment',
    'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism',
    'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral'
    ]

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

from sklearn.feature_extraction.text import TfidfVectorizer

y_train_baseline = train_df['labels'].apply(lambda x: x[0])
y_val_baseline = validation_df['labels'].apply(lambda x: x[0])
y_test_baseline = test_df['labels'].apply(lambda x: x[0])

tfidf_vectorizer = TfidfVectorizer()

# Fit on training data and transform all splits
X_train_baseline = tfidf_vectorizer.fit_transform(train_df['text'])
X_val_baseline = tfidf_vectorizer.transform(validation_df['text'])
X_test_baseline = tfidf_vectorizer.transform(test_df['text'])

print("Shape of baseline features - Train:", X_train_baseline.shape)
print("Shape of baseline features - Validation:", X_val_baseline.shape)
print("Shape of baseline features - Test:", X_test_baseline.shape)

# Define models with their default settings
models = {
    'SVC': SVC(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': GradientBoostingClassifier()
}

results = []

for name, model in models.items():
    # Train the model
    model.fit(X_train_baseline, y_train_baseline)
    
    # Make predictions
    train_pred = model.predict(X_train_baseline)
    val_pred = model.predict(X_val_baseline)
    test_pred = model.predict(X_test_baseline)
    
    # Calculate accuracies
    train_acc = accuracy_score(y_train_baseline, train_pred)
    val_acc = accuracy_score(y_val_baseline, val_pred)
    test_acc = accuracy_score(y_test_baseline, test_pred)
    
    # Store results
    results.append({
        'Model': name,
        'Train Accuracy': f'{train_acc:.4f}',
        'Validation Accuracy': f'{val_acc:.4f}',
        'Test Accuracy': f'{test_acc:.4f}',
    })
    
    print(name,str(model.get_params()))

# Create a DataFrame from the results
df_results = pd.DataFrame(results)

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Hide axes
ax.axis('off')

# Create the table
table = ax.table(cellText=df_results.values, colLabels=df_results.columns, cellLoc='center', loc='center')

# Set font size
table.auto_set_font_size(False)
table.set_fontsize(8)

# Scale the table to fit the figure
table.scale(1, 1.5)

# Save the table as an image
plt.savefig('model_comparison_table.png', dpi=300, bbox_inches='tight')
plt.close()

print("Model comparison table has been saved as 'model_comparison_table.png'")

# Print the results
print(df_results.to_string(index=False))
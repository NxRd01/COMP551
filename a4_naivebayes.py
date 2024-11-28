from datasets import load_dataset
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import matplotlib.pyplot as plt

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

from sklearn.feature_extraction.text import CountVectorizer

count_vectorizer = CountVectorizer()

# Fit on training data and transform all splits
X_train_nb = count_vectorizer.fit_transform(train_df['text'])
X_val_nb = count_vectorizer.transform(validation_df['text'])
X_test_nb = count_vectorizer.transform(test_df['text'])

y_train = train_df['labels'].apply(lambda x: x[0])
y_val = validation_df['labels'].apply(lambda x: x[0])
y_test = test_df['labels'].apply(lambda x: x[0])

print("Shape of Naive Bayes features - Train:", X_train_nb.shape)
print("Shape of Naive Bayes features - Validation:", X_val_nb.shape)
print("Shape of Naive Bayes features - Test:", X_test_nb.shape)

class NaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # Smoothing parameter
        self.classes = None
        self.class_priors = None
        self.feature_probs = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        n_classes = len(self.classes)

        # Calculate class priors
        self.class_priors = np.zeros(n_classes)
        for i, c in enumerate(self.classes):
            self.class_priors[i] = np.sum(y == c) / n_samples

        # Calculate feature probabilities
        self.feature_probs = np.zeros((n_classes, n_features))
        for i, c in enumerate(self.classes):
            X_c = X[y == c]
            self.feature_probs[i] = (np.sum(X_c, axis=0) + self.alpha) / (np.sum(X_c) + self.alpha * n_features)

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        log_probs = np.log(self.class_priors) + np.sum(np.log(self.feature_probs) * x, axis=1)
        return self.classes[np.argmax(log_probs)]

    def evaluate_acc(self, y_true, y_pred):
        return np.mean(y_true == y_pred)
    
    
    

X_train_nb = X_train_nb.toarray()  # Convert sparse matrix to dense array
X_val_nb = X_val_nb.toarray()
X_test_nb = X_test_nb.toarray()




def plot_learning_curve(X_train, y_train, X_val, y_val):
    alphas = np.logspace(-4, 3, 20)
    train_scores = []
    val_scores = []

    best_alpha = None
    best_val_score = 0

    for alpha in alphas:
        model = NaiveBayes(alpha=alpha)
        model.fit(X_train, y_train)
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        train_score = model.evaluate_acc(y_train, train_pred)
        val_score = model.evaluate_acc(y_val, val_pred)
        train_scores.append(train_score)
        val_scores.append(val_score)

        if val_score > best_val_score:
            best_val_score = val_score
            best_alpha = alpha

    plt.figure(figsize=(10, 6))
    plt.semilogx(alphas, train_scores, label='Training Score')
    plt.semilogx(alphas, val_scores, label='Validation Score')
    plt.xlabel('Alpha')
    plt.ylabel('Accuracy')
    plt.title('Naive Bayes Learning Curve')
    plt.legend()
    plt.grid(True)

    # Save the plot as an image
    plt.savefig('naive_bayes_learning_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Best alpha: {best_alpha:.6f}")
    print(f"Best validation score: {best_val_score:.4f}")

    return best_alpha


best_alpha = plot_learning_curve(X_train_nb, y_train, X_val_nb, y_val)

# Now use the best alpha to train the final model and evaluate on test set
final_model = NaiveBayes(alpha=best_alpha)
final_model.fit(X_train_nb, y_train)
test_pred = final_model.predict(X_test_nb)
test_score = final_model.evaluate_acc(y_test, test_pred)
print(f"Test score with best alpha: {test_score:.4f}")
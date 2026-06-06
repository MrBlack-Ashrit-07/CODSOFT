# Importing Libraries
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
import os
dataset_path = os.path.join('Genre Classification Dataset', 'train_data.txt')
df = pd.read_csv(
    dataset_path,
    sep=' ::: ',
    engine='python',
    names=['ID', 'Title', 'Genre', 'Description']
)
print(df.head())

# Prepare Data
X = df['Description']
y = df['Genre']
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
)

# Create Pipeline
model = Pipeline([
    ('tdidf', TfidfVectorizer(stop_words = 'english')),
    ('classifier', MultinomialNB())
])

# Train Model
model.fit(X_train, y_train)

# Predictions on Test Data
y_pred = model.predict(X_test)

# Accuracy Score
print('Accuracy', accuracy_score(y_test, y_pred))
print("\nClassification Report: ")
print(classification_report(y_test, y_pred))
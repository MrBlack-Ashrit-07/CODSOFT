import pandas as pd
import numpy as np
import os
import re
import time
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Define standard English stopwords to avoid external package download dependencies
STOPWORDS = set([
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
    "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", 
    "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", 
    "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", 
    "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", 
    "for", "with", "about", "against", "between", "into", "through", "during", "before", 
    "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", 
    "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", 
    "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", 
    "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", 
    "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve", "y"
])

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase the text
    text = text.lower()
    # Remove special characters and numbers
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize and remove stopwords
    words = [word for word in text.split() if word not in STOPWORDS]
    # Re-join words
    return " ".join(words)

def main():
    start_time = time.time()
    
    # Paths relative to the script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(os.path.dirname(script_dir), 'Genre Classification Dataset')
    train_path = os.path.join(dataset_dir, 'train_data.txt')
    
    print(f"Loading training data from: {train_path}...")
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found!")
        return
        
    # Load dataset
    df = pd.read_csv(
        train_path,
        sep=' ::: ',
        engine='python',
        names=['ID', 'Title', 'Genre', 'Description']
    )
    
    print(f"Loaded {len(df)} records.")
    print("Class distribution of genres (Top 10):")
    print(df['Genre'].value_counts().head(10))
    print("-" * 50)
    
    # Preprocessing
    print("Cleaning text descriptions (removing punctuation, lowercase, stop words)...")
    df['Cleaned_Description'] = df['Description'].apply(clean_text)
    print("Preprocessing completed.")
    
    # Split training and validation sets (80-20 split)
    print("Splitting dataset into train and validation sets...")
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        df['Cleaned_Description'], 
        df['Genre'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['Genre']  # ensure class distributions are preserved
    )
    
    # Feature extraction - TF-IDF
    print("Extracting TF-IDF features (max_features=20000, ngram_range=(1,2))...")
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_raw)
    X_val = vectorizer.transform(X_val_raw)
    
    # Define models
    models = {
        'Multinomial Naive Bayes': MultinomialNB(),
        'Linear SVC': LinearSVC(class_weight='balanced', random_state=42, max_iter=2000, dual=False)
    }
    
    best_val_acc = 0
    best_model_name = None
    
    print("\nTraining and evaluating models on validation set...")
    print("=" * 60)
    
    for name, clf in models.items():
        print(f"Training {name}...")
        t0 = time.time()
        clf.fit(X_train, y_train)
        train_time = time.time() - t0
        
        # Predict on validation set
        y_pred = clf.predict(X_val)
        
        # Calculate metrics
        acc = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average='macro')
        
        print(f"[{name}] Training Time: {train_time:.2f}s | Validation Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
        
        # Keep track of the best model based on accuracy
        if acc > best_val_acc:
            best_val_acc = acc
            best_model_name = name
            
    print("=" * 60)
    print(f"Best model based on validation accuracy: {best_model_name} ({best_val_acc:.4f})")
    
    # Train the winning model on the full training dataset
    print(f"\nRe-training best model ({best_model_name}) on full training dataset...")
    X_full = vectorizer.fit_transform(df['Cleaned_Description'])
    y_full = df['Genre']
    
    winning_model = models[best_model_name]
    winning_model.fit(X_full, y_full)
    
    # Save the pipeline components
    model_output_path = os.path.join(script_dir, 'model.joblib')
    print(f"Saving vectorizer and model to: {model_output_path}...")
    joblib.dump({
        'vectorizer': vectorizer,
        'model': winning_model,
        'model_name': best_model_name
    }, model_output_path)
    
    total_time = time.time() - start_time
    print(f"\nSuccess! Entire process finished in {total_time:.2f} seconds.")

if __name__ == '__main__':
    main()

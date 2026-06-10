import re
import os
import time
import joblib
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, f1_score

# Define standard English stopwords to clean document texts
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
    # Lowercase
    text = text.lower()
    # Remove email addresses pattern
    text = re.sub(r'\S+@\S+', '', text)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize and remove stopwords and short words
    words = [word for word in text.split() if word not in STOPWORDS and len(word) > 1]
    return " ".join(words)

def main():
    start_time = time.time()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    print("Fetching 20 Newsgroups training dataset (removing headers, footers, quotes)...")
    newsgroups_train = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    
    X_raw = newsgroups_train.data
    y = newsgroups_train.target
    target_names = newsgroups_train.target_names
    
    print(f"Loaded {len(X_raw)} documents.")
    print(f"Categories ({len(target_names)} classes):")
    print(", ".join(target_names))
    print("-" * 60)
    
    # Preprocess text
    print("Preprocessing and cleaning documents...")
    X_cleaned = [clean_text(doc) for doc in X_raw]
    
    # Split into train & validation (80-20)
    print("Splitting into train and validation sets...")
    X_train_raw, X_val_raw, y_train, y_val = train_test_split(
        X_cleaned, 
        y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    # TF-IDF Vectorization
    print("Extracting TF-IDF features (max_features=25000, ngram_range=(1,2))...")
    vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 2))
    X_train = vectorizer.fit_transform(X_train_raw)
    X_val = vectorizer.transform(X_val_raw)
    
    # Define models
    models = {
        'Multinomial Naive Bayes': MultinomialNB(alpha=0.1),
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
        
        y_pred = clf.predict(X_val)
        acc = accuracy_score(y_val, y_pred)
        macro_f1 = f1_score(y_val, y_pred, average='macro')
        
        print(f"[{name}] Training Time: {train_time:.2f}s | Validation Accuracy: {acc:.4f} | Macro F1: {macro_f1:.4f}")
        
        if acc > best_val_acc:
            best_val_acc = acc
            best_model_name = name
            
    print("=" * 60)
    print(f"Best model based on validation accuracy: {best_model_name} ({best_val_acc:.4f})")
    
    # Train the winning model on the entire training set
    print(f"\nRe-training best model ({best_model_name}) on full training subset...")
    X_full = vectorizer.fit_transform(X_cleaned)
    
    winning_model = models[best_model_name]
    winning_model.fit(X_full, y)
    
    # Save the pipeline
    model_output_path = os.path.join(script_dir, 'doc_model.joblib')
    print(f"Saving vectorizer and model to: {model_output_path}...")
    joblib.dump({
        'vectorizer': vectorizer,
        'model': winning_model,
        'model_name': best_model_name,
        'target_names': target_names
    }, model_output_path)
    
    total_time = time.time() - start_time
    print(f"\nSuccess! Training completed in {total_time:.2f} seconds.")

if __name__ == '__main__':
    main()

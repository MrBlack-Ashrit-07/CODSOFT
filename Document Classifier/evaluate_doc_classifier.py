import time
import os
import sys
import joblib
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Add current script's directory to sys.path to ensure we can import train_doc_classifier
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from train_doc_classifier import clean_text

def main():
    start_time = time.time()
    
    # Load Model
    model_path = os.path.join(script_dir, 'doc_model.joblib')
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found! Please run train_doc_classifier.py first.")
        return
        
    print(f"Loading model and vectorizer from: {model_path}...")
    saved_data = joblib.load(model_path)
    vectorizer = saved_data['vectorizer']
    model = saved_data['model']
    model_name = saved_data.get('model_name', 'Trained Classifier')
    target_names = saved_data['target_names']
    print(f"Loaded {model_name} successfully.")
    
    # Fetch Test Dataset
    print("\nFetching 20 Newsgroups test dataset (removing headers, footers, quotes)...")
    newsgroups_test = fetch_20newsgroups(subset='test', remove=('headers', 'footers', 'quotes'))
    
    X_raw = newsgroups_test.data
    y_test = newsgroups_test.target
    
    print(f"Loaded {len(X_raw)} test documents.")
    
    # Preprocess test set
    print("Preprocessing test documents...")
    X_test_cleaned = [clean_text(doc) for doc in X_raw]
    
    # Vectorize
    print("Extracting TF-IDF features from test documents...")
    X_test = vectorizer.transform(X_test_cleaned)
    
    # Inference
    print(f"Making predictions on the test set using {model_name}...")
    t0 = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - t0
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred, target_names=target_names)
    
    print("\n" + "=" * 60)
    print(f"TEST PERFORMANCE REPORT ({model_name}) - 20 Newsgroups")
    print("=" * 60)
    print(f"Inference Time for {len(X_raw)} documents: {inference_time:.2f}s")
    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test Macro F1 : {macro_f1:.4f}")
    print("\nDetailed Classification Report:")
    print(report)
    print("=" * 60)
    
    # Save test report
    report_output_path = os.path.join(script_dir, 'doc_test_evaluation_report.txt')
    print(f"Saving test report to {report_output_path}...")
    with open(report_output_path, 'w') as f:
        f.write(f"TEST PERFORMANCE REPORT ({model_name}) - 20 Newsgroups\n")
        f.write("=" * 60 + "\n")
        f.write(f"Test Accuracy : {acc:.4f}\n")
        f.write(f"Test Macro F1 : {macro_f1:.4f}\n")
        f.write(f"Inference Time: {inference_time:.2f}s\n")
        f.write("=" * 60 + "\n\n")
        f.write(report)
        
    total_time = time.time() - start_time
    print(f"Evaluation completed in {total_time:.2f} seconds.")

if __name__ == '__main__':
    main()

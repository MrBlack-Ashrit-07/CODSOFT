import pandas as pd
import numpy as np
import os
import sys
import time
import joblib
from sklearn.metrics import accuracy_score, classification_report, f1_score

# Add current script's directory to sys.path to ensure we can import train
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from train import clean_text

def main():
    start_time = time.time()
    
    # Model Path relative to script directory
    model_path = os.path.join(script_dir, 'model.joblib')
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found! Please run train.py first to train the model.")
        return
        
    print(f"Loading model and vectorizer from: {model_path}...")
    saved_data = joblib.load(model_path)
    vectorizer = saved_data['vectorizer']
    model = saved_data['model']
    model_name = saved_data.get('model_name', 'Trained Classifier')
    print(f"Loaded {model_name} successfully.")
    
    # Test Data Path relative to parent of script directory
    dataset_dir = os.path.join(os.path.dirname(script_dir), 'Genre Classification Dataset')
    test_solution_path = os.path.join(dataset_dir, 'test_data_solution.txt')
    
    print(f"Loading test data solution from: {test_solution_path}...")
    if not os.path.exists(test_solution_path):
        print(f"Error: {test_solution_path} not found!")
        return
        
    # Load dataset
    df_test = pd.read_csv(
        test_solution_path,
        sep=' ::: ',
        engine='python',
        names=['ID', 'Title', 'Genre', 'Description']
    )
    
    print(f"Loaded {len(df_test)} test records.")
    
    # Preprocessing
    print("Preprocessing test descriptions...")
    df_test['Cleaned_Description'] = df_test['Description'].apply(clean_text)
    print("Preprocessing completed.")
    
    # Feature extraction
    print("Extracting TF-IDF features from test data...")
    X_test = vectorizer.transform(df_test['Cleaned_Description'])
    y_test = df_test['Genre']
    
    # Inference
    print(f"Making predictions on the test set using {model_name}...")
    t0 = time.time()
    y_pred = model.predict(X_test)
    inference_time = time.time() - t0
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    report = classification_report(y_test, y_pred)
    
    print("\n" + "=" * 60)
    print(f"TEST PERFORMANCE REPORT ({model_name})")
    print("=" * 60)
    print(f"Inference Time for {len(df_test)} records: {inference_time:.2f}s")
    print(f"Test Accuracy : {acc:.4f}")
    print(f"Test Macro F1 : {macro_f1:.4f}")
    print("\nDetailed Classification Report:")
    print(report)
    print("=" * 60)
    
    # Save the report to a text file
    report_output_path = os.path.join(script_dir, 'test_evaluation_report.txt')
    print(f"Saving test report to {report_output_path}...")
    with open(report_output_path, 'w') as f:
        f.write(f"TEST PERFORMANCE REPORT ({model_name})\n")
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

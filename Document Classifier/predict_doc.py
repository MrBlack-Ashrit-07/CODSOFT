import sys
import os
import joblib
import numpy as np

# Add current script's directory to sys.path to ensure we can import train_doc_classifier
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from train_doc_classifier import clean_text

def get_probabilities(model, X_features):
    """
    Get probability/confidence estimates for model predictions.
    If MultinomialNB, uses predict_proba.
    If LinearSVC, computes softmax over decision_function scores.
    Handles both binary and multiclass classifiers.
    """
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X_features)[0]
    elif hasattr(model, 'decision_function'):
        scores = model.decision_function(X_features)[0]
        # Handle binary vs multiclass decision function shape
        if np.ndim(scores) == 0 or len(np.atleast_1d(scores)) == 1:
            s = np.atleast_1d(scores)[0]
            prob_positive = 1.0 / (1.0 + np.exp(-s))
            return np.array([1.0 - prob_positive, prob_positive])
        else:
            exp_scores = np.exp(scores - np.max(scores))
            return exp_scores / exp_scores.sum()
    else:
        return None

def predict_document_category(doc_text, vectorizer, model, target_names):
    cleaned = clean_text(doc_text)
    if not cleaned.strip():
        print("Warning: Cleaned document is empty! Please write some meaningful text.")
        return
        
    X_features = vectorizer.transform([cleaned])
    
    pred_idx = model.predict(X_features)[0]
    pred_category = target_names[pred_idx]
    
    probs = get_probabilities(model, X_features)
    
    print("\n" + "=" * 50)
    print("INPUT DOCUMENT:")
    print(doc_text[:150] + ("..." if len(doc_text) > 150 else ""))
    print("-" * 50)
    print(f"PREDICTED CATEGORY: {pred_category.upper()}")
    
    if probs is not None:
        print("\nTOP 3 CONFIDENCE ESTIMATES:")
        top_indices = np.argsort(probs)[::-1][:3]
        for idx in top_indices:
            # Map index in probability array to the actual class label (integer)
            class_label = model.classes_[idx]
            cat_name = target_names[class_label]
            prob = probs[idx]
            print(f"  - {cat_name:<25}: {prob * 100:.2f}%")
    print("=" * 50)

def main():
    model_path = os.path.join(script_dir, 'doc_model.joblib')
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found! Please run train_doc_classifier.py first.")
        return
        
    saved_data = joblib.load(model_path)
    vectorizer = saved_data['vectorizer']
    model = saved_data['model']
    model_name = saved_data.get('model_name', 'Trained Classifier')
    target_names = saved_data['target_names']
    
    if len(sys.argv) > 1:
        doc_text = " ".join(sys.argv[1:])
        predict_document_category(doc_text, vectorizer, model, target_names)
    else:
        print(f"=== Document Category Predictor (Using {model_name}) ===")
        print("Enter document content or snippet to predict its category. Type 'exit' or 'quit' to stop.\n")
        
        while True:
            try:
                doc_text = input("Document Snippet: ").strip()
                if not doc_text:
                    continue
                if doc_text.lower() in ['exit', 'quit']:
                    print("Exiting. Goodbye!")
                    break
                predict_document_category(doc_text, vectorizer, model, target_names)
                print()
            except KeyboardInterrupt:
                print("\nExiting. Goodbye!")
                break
            except Exception as e:
                print(f"Error making prediction: {e}")

if __name__ == '__main__':
    main()

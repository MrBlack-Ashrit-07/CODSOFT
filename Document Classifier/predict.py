import sys
import os
import joblib
import numpy as np

# Add current script's directory to sys.path to ensure we can import train
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)
from train import clean_text

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
            exp_scores = np.exp(scores - np.max(scores)) # subtract max for numerical stability
            return exp_scores / exp_scores.sum()
    else:
        return None

def predict_genre(description, vectorizer, model, classes):
    # Preprocess description
    cleaned = clean_text(description)
    
    # Vectorize
    X_features = vectorizer.transform([cleaned])
    
    # Predict
    pred_genre = model.predict(X_features)[0]
    
    # Probabilities
    probs = get_probabilities(model, X_features)
    
    print("\n" + "=" * 50)
    print("INPUT DESCRIPTION:")
    print(description[:150] + ("..." if len(description) > 150 else ""))
    print("-" * 50)
    print(f"PREDICTED GENRE: {pred_genre.upper()}")
    
    if probs is not None:
        print("\nTOP 3 CONFIDENCE ESTIMATES:")
        # Get indices of top 3 probabilities
        top_indices = np.argsort(probs)[::-1][:3]
        for idx in top_indices:
            genre = classes[idx]
            prob = probs[idx]
            print(f"  - {genre:<15}: {prob * 100:.2f}%")
    print("=" * 50)

def main():
    model_path = os.path.join(script_dir, 'model.joblib')
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found! Please run train.py first.")
        return
        
    saved_data = joblib.load(model_path)
    vectorizer = saved_data['vectorizer']
    model = saved_data['model']
    model_name = saved_data.get('model_name', 'Trained Classifier')
    
    classes = model.classes_
    
    # If arguments are passed, predict on the arguments
    if len(sys.argv) > 1:
        description = " ".join(sys.argv[1:])
        predict_genre(description, vectorizer, model, classes)
    else:
        # Interactive mode
        print(f"=== Movie Genre Predictor (Using {model_name}) ===")
        print("Enter a movie description to predict its genre. Type 'exit' or 'quit' to stop.\n")
        
        while True:
            try:
                description = input("Movie Description: ").strip()
                if not description:
                    continue
                if description.lower() in ['exit', 'quit']:
                    print("Exiting. Goodbye!")
                    break
                predict_genre(description, vectorizer, model, classes)
                print()
            except KeyboardInterrupt:
                print("\nExiting. Goodbye!")
                break
            except Exception as e:
                print(f"Error making prediction: {e}")

if __name__ == '__main__':
    main()

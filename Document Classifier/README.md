# CODSOFT 20 Newsgroups Document Classifier

An end-to-end Machine Learning project that classifies text documents into their standard newsgroup categories using TF-IDF features and a Multinomial Naive Bayes classifier.

This project trains on scikit-learn's built-in `20 Newsgroups` dataset, containing ~18,000 documents across 20 distinct categories.

## Project Structure
*   **`train_doc_classifier.py`**: Fetches the 20 Newsgroups training dataset, cleans and preprocesses the text, compares classifiers (Naive Bayes vs. LinearSVC), and saves the best model to `doc_model.joblib`.
*   **`evaluate_doc_classifier.py`**: Loads the saved model and evaluates it against the official test subset, outputting accuracy and a detailed classification report.
*   **`predict_doc.py`**: Runs inference on custom document snippets. Supports both interactive prompt mode and passing text via CLI arguments.

## Setup & Dependencies
Ensure you have the required Python packages installed:
```bash
pip install pandas numpy scikit-learn joblib
```

## Running the Project

### 1. Train the Classifier
Run `train_doc_classifier.py` to train and save the model:
```bash
python train_doc_classifier.py
```

### 2. Evaluate on Test Dataset
Evaluate the trained classifier's metrics against the official test set:
```bash
python evaluate_doc_classifier.py
```
This generates a detailed classification report saved in `doc_test_evaluation_report.txt`.

### 3. Predict Custom Document Categories
You can run predictions in two ways:

#### A. Command Line Arguments
```bash
python predict_doc.py "The Hubble telescope captured a stunning image of a distant spiral galaxy containing billions of stars."
```

#### B. Interactive Terminal Mode
Start the script without arguments to enter an interactive prompt loop:
```bash
python predict_doc.py
```
Type `exit` or `quit` to stop the interactive session.
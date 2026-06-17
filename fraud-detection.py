# Credit Card Fraud Detection Pipeline
# Author: Antigravity AI Coding Assistant

import pandas as pd
import numpy as np
import os
import time
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    roc_curve, precision_recall_curve
)
import matplotlib.pyplot as plt

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth 
    (specified in decimal degrees) using the Haversine formula.
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    r = 6371.0  # Radius of Earth in kilometers
    return c * r

def preprocess_data(df):
    """
    Applies feature engineering to the raw dataset and selects relevant columns.
    """
    print("  Converting date columns to datetime objects...")
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    
    print("  Calculating cardholder age...")
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days / 365.25
    
    print("  Extracting temporal features (hour, day of week, month)...")
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['month'] = df['trans_date_trans_time'].dt.month
    
    print("  Calculating geodesic distance between customer and merchant...")
    df['distance'] = haversine_distance(
        df['lat'], df['long'], df['merch_lat'], df['merch_long']
    )
    
    print("  Encoding gender...")
    df['gender'] = df['gender'].map({'F': 0, 'M': 1})
    df['gender'] = df['gender'].fillna(0).astype(int)
    
    # Feature columns to retain for training
    feature_cols = [
        'amt', 'distance', 'age', 'city_pop',
        'hour', 'day_of_week', 'month', 'gender', 'category'
    ]
    
    return df[feature_cols].copy()

def main():
    print("==================================================")
    print("Starting Credit Card Fraud Detection Pipeline")
    print("==================================================")
    
    # Define columns to load
    cols_to_load = [
        'trans_date_trans_time', 'category', 'amt', 'gender', 'lat', 'long',
        'city_pop', 'dob', 'merch_lat', 'merch_long', 'is_fraud'
    ]
    
    # 1. Load and downsample training dataset
    print("\n[Step 1/6] Loading training data...")
    if not os.path.exists('fraudTrain.csv'):
        raise FileNotFoundError("Error: 'fraudTrain.csv' not found in the current directory.")
        
    train_df = pd.read_csv('fraudTrain.csv', usecols=cols_to_load)
    print(f"  Total training records: {len(train_df):,}")
    
    # Downsample to address severe class imbalance and speed up training
    fraud_train = train_df[train_df['is_fraud'] == 1]
    non_fraud_train = train_df[train_df['is_fraud'] == 0]
    
    print(f"  Class breakdown - Legitimate: {len(non_fraud_train):,}, Fraudulent: {len(fraud_train):,}")
    
    legit_sample_size = 150000
    print(f"  Downsampling legitimate transactions to {legit_sample_size:,} samples...")
    non_fraud_sampled = non_fraud_train.sample(n=legit_sample_size, random_state=42)
    
    # Concatenate and shuffle
    train_sampled = pd.concat([fraud_train, non_fraud_sampled]).sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"  Sampled training dataset size: {len(train_sampled):,}")
    
    # 2. Load the test dataset
    print("\n[Step 2/6] Loading test data...")
    if not os.path.exists('fraudTest.csv'):
        raise FileNotFoundError("Error: 'fraudTest.csv' not found in the current directory.")
        
    test_df = pd.read_csv('fraudTest.csv', usecols=cols_to_load)
    print(f"  Total test records (unmodified/realistic distribution): {len(test_df):,}")
    print(f"  Test class breakdown - Legitimate: {len(test_df[test_df['is_fraud'] == 0]):,}, Fraudulent: {len(test_df[test_df['is_fraud'] == 1]):,}")
    
    # 3. Perform feature engineering
    print("\n[Step 3/6] Applying feature engineering on train data...")
    X_train_raw = preprocess_data(train_sampled)
    y_train = train_sampled['is_fraud'].values
    
    print("Applying feature engineering on test data...")
    X_test_raw = preprocess_data(test_df)
    y_test = test_df['is_fraud'].values
    
    # 4. Fit column transformer (Scaling + One-Hot Encoding)
    print("\n[Step 4/6] Preprocessing features using ColumnTransformer...")
    numeric_cols = ['amt', 'distance', 'age', 'city_pop']
    categorical_cols = ['category']
    # 'hour', 'day_of_week', 'month', 'gender' will remain as passthrough
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
        ],
        remainder='passthrough'
    )
    
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    print(f"  Transformed feature shape: {X_train.shape}")
    
    # 5. Initialize and train models
    print("\n[Step 5/6] Training classifiers...")
    
    models = {
        'Logistic Regression': LogisticRegression(
            class_weight='balanced', 
            max_iter=1000, 
            random_state=42
        ),
        'Decision Tree': DecisionTreeClassifier(
            class_weight='balanced', 
            max_depth=12, 
            random_state=42
        ),
        'Random Forest': RandomForestClassifier(
            class_weight='balanced', 
            n_estimators=100, 
            max_depth=12, 
            random_state=42, 
            n_jobs=-1
        )
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- Training {name} ---")
        start = time.time()
        model.fit(X_train, y_train)
        duration = time.time() - start
        print(f"  Completed in {duration:.2f} seconds.")
        
        print(f"  Predicting on test data...")
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        
        results[name] = {
            'Accuracy': acc,
            'Precision': prec,
            'Recall': rec,
            'F1-Score': f1,
            'ROC-AUC': roc_auc,
            'PR-AUC': pr_auc,
            'Train-Time': duration,
            'y_prob': y_prob
        }
        
        print(f"\nClassification Report for {name}:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraudulent']))
        print(f"F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}")
        print("-" * 50)
        
    # 6. Compare and visualize results
    print("\n[Step 6/6] Formatting comparisons and generating plots...")
    
    # Print summary table
    summary_data = []
    for name, res in results.items():
        summary_data.append({
            'Model': name,
            'Accuracy': f"{res['Accuracy']:.3%}",
            'Precision': f"{res['Precision']:.3%}",
            'Recall': f"{res['Recall']:.3%}",
            'F1-Score': f"{res['F1-Score']:.4f}",
            'ROC-AUC': f"{res['ROC-AUC']:.4f}",
            'PR-AUC': f"{res['PR-AUC']:.4f}",
            'Train Time (s)': f"{res['Train-Time']:.2f}"
        })
    summary_df = pd.DataFrame(summary_data)
    print("\n" + "=" * 32 + " MODEL COMPARISON SUMMARY " + "=" * 32)
    print(summary_df.to_string(index=False))
    print("=" * 90)
    
    # Plot curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    
    # Plot ROC Curves
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        ax1.plot(fpr, tpr, label=f"{name} (ROC-AUC = {res['ROC-AUC']:.4f})", linewidth=2)
    ax1.plot([0, 1], [0, 1], 'k--', label='Random Guess', alpha=0.5)
    ax1.set_xlabel('False Positive Rate', fontsize=12)
    ax1.set_ylabel('True Positive Rate', fontsize=12)
    ax1.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='lower right', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Plot Precision-Recall Curves
    for name, res in results.items():
        precision, recall, _ = precision_recall_curve(y_test, res['y_prob'])
        ax2.plot(recall, precision, label=f"{name} (PR-AUC = {res['PR-AUC']:.4f})", linewidth=2)
    ax2.set_xlabel('Recall', fontsize=12)
    ax2.set_ylabel('Precision', fontsize=12)
    ax2.set_title('Precision-Recall Curves (Key for Imbalanced Data)', fontsize=14, fontweight='bold')
    ax2.legend(loc='lower left', fontsize=10)
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    plt.suptitle('Performance Comparison for Credit Card Fraud Detection Models', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig('metrics_comparison.png', dpi=300)
    print("\nComparison plot saved successfully as 'metrics_comparison.png'")
    print("Pipeline completed successfully!")

if __name__ == '__main__':
    main()

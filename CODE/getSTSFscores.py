import pandas as pd
import os
import numpy as np
from tsc_datasets import univariate_equal_length
from data_loader import load_datasets
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder

metrics = ['accuracy', 'trtime', 'f1', 'balanced_accuracy', 'auc']

def evaluate_metric(y_test, y_pred, y_preds_proba, metric, cr):
    """
    Evaluate a specific metric for the classifier.
    
    Parameters:
    - y_true: The ground truth labels.
    - y_pred: The predicted labels.
    - metric: The metric to calculate (e.g., 'accuracy', 'f1', 'balanced_accuracy', 'auc').
    
    Returns:
    - The value of the requested metric.
    """
    # Create a LabelEncoder object
    le = LabelEncoder()

    # Fit and transform y_test to convert string labels to numeric
    y_test_encoded = le.fit_transform(y_test)
    y_pred = y_pred.astype(int)
    if metric == 'accuracy':
        return accuracy_score(y_test_encoded, y_pred)
    elif metric == 'trtime':
        return cr.fit_time
    elif metric == 'f1':
        return f1_score(y_test_encoded, y_pred, average='weighted')
    elif metric == 'balanced_accuracy':
        return balanced_accuracy_score(y_test_encoded, y_pred)
    elif metric == 'auc':
        # AUC can only be calculated for binary classification with probabilities
        return roc_auc_score(y_test_encoded, y_preds_proba, average='weighted', multi_class='ovo') if len(set(y_test_encoded)) > 2 else roc_auc_score(y_test_encoded, y_preds_proba[:,1])
    else:
        raise ValueError("Unsupported metric. Choose from: 'accuracy', 'f1', 'balanced_accuracy', 'auc'.")


path = "C:/Users/BOUKA/OneDrive/Bureau/CODE/tsml_eval/SupervisedTimeSeriesForest/Predictions/"

datasets_list = list(univariate_equal_length)
datasets_dict = load_datasets(datasets_list)

STSF_results = {}
from tsml_eval.evaluation.storage import load_classifier_results

for dataset_name, data in datasets_dict.items():
    # Load the training and test data
    X_train, y_train = data['X_train'], data['y_train']
    X_test, y_test = data['X_test'], data['y_test']
    preds_path = ""

    y_pred_path = f"{path}{dataset_name}/testResample0.csv"
    STSF_results[dataset_name]={}

    cr=load_classifier_results(y_pred_path)
    y_pred = cr.predictions
    y_proba = cr.probabilities
    
    for metric in metrics:
        STSF_results[dataset_name][metric] = evaluate_metric(y_test, y_pred, y_proba, metric,cr)


# Convert dictionary to DataFrame
df = pd.DataFrame.from_dict(STSF_results, orient='index')

# Reset the index to make datasets a column
df.reset_index(inplace=True)

# Rename the index column to 'dataset'
df.rename(columns={'index': 'dataset'}, inplace=True)

#  save as Excel
df.to_excel('STSF_results.xlsx', index=False)


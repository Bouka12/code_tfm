import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, roc_auc_score
metrics = ['accuracy', 'f1', 'balanced_accuracy', 'auc']

# Evaluation function for a specific metric
def evaluate_metric(y_true, y_pred, y_preds_proba, metric):
    """
    Evaluate a specific metric for the classifier.
    
    Parameters:
    - y_true: The ground truth labels.
    - y_pred: The predicted labels.
    - metric: The metric to calculate (e.g., 'accuracy', 'f1', 'balanced_accuracy', 'auc').
    
    Returns:
    - The value of the requested metric.
    """
    if metric == 'accuracy':
        return accuracy_score(y_true, y_pred)
    elif metric == 'f1':
        return f1_score(y_true, y_pred, average='weighted')
    elif metric == 'balanced_accuracy':
        return balanced_accuracy_score(y_true, y_pred)
    elif metric == 'auc':
        # AUC can only be calculated for binary classification with probabilities
        return roc_auc_score(y_true, y_preds_proba, average='weighted', multi_class='ovo') if len(set(y_true)) > 2 else roc_auc_score(y_true, y_preds_proba[:,1])
    else:
        raise ValueError("Unsupported metric. Choose from: 'accuracy', 'f1', 'balanced_accuracy', 'auc'.")

# Function to save results for one metric across all datasets
def save_results_for_metric(data_dict, preds_proba_dict, preds_dict, metric, experiment):
    """
    Evaluate a specific metric for multiple datasets and save the results in an Excel file.
    
    Parameters:
    - data_dict: Dictionary where each key is a dataset name, and the value is a tuple of (y_true, y_pred).
    - metric: The metric to calculate and save (e.g., 'accuracy', 'f1', 'balanced_accuracy', 'auc').
    """
    # List to store results for all datasets
    results = []

    for dataset_name, data in data_dict.items():
        y_true, y_pred, y_preds_proba = data['y_test'], preds_dict[dataset_name], preds_proba_dict[dataset_name]

        metric_value = evaluate_metric(y_true, y_pred,y_preds_proba, metric)
        results.append({'Dataset': dataset_name, metric.capitalize(): metric_value})

    # Convert the results to a DataFrame
    results_df = pd.DataFrame(results)

    # Get the script name to use for naming the Excel file
    script_name = experiment  # Get the script file name without extension
    # Define the results directory and ensure it exists
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Create the Excel file name based on the script and metric
    file_name = f"{script_name}_{metric}.xlsx"
    file_path = os.path.join(results_dir, file_name)

    # Save the results to the Excel file in the 'results/' directory
    results_df.to_excel(file_path, index=False)

    print(f"Results for {metric} saved to {file_path}")

# Example usage for multiple datasets
def evaluate_datasets(data_dict, preds_proba_dict, preds_dict, metrics, experiment):
    """
    Evaluate multiple datasets for a specific metric and save the results in an Excel file.
    
    Parameters:
    - data_dict: Dictionary where each key is a dataset name, and the value is a tuple of (y_true, y_pred).
    - metric: The metric to calculate and save (e.g., 'accuracy', 'f1', 'balanced_accuracy', 'auc').
    """
    for metric in metrics:
        save_results_for_metric(data_dict,preds_proba_dict, preds_dict, metric, experiment)



# evaluate_datasets(data_dict, 'accuracy')
# evaluate_datasets(data_dict, 'f1')
# evaluate_datasets(data_dict, 'balanced_accuracy')
# evaluate_datasets(data_dict, 'auc')

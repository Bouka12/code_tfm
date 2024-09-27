import pandas as pd
import os
import scikit_posthocs as sp
import scipy.stats as ss
import matplotlib.pyplot as plt
import numpy as np
from aeon.visualisation import plot_critical_difference

save_dir = "C:/Users/BOUKA/OneDrive/Bureau/CODE/plots/" # C:\Users\BOUKA\OneDrive\Bureau\CODE\plots
os.makedirs(save_dir, exist_ok=True)

# Function to filter the dataframe based on the metric
def filter_by_metric(df, metric):
    """
    Filter the DataFrame to only include columns related to the specified metric (e.g., "Accuracy", "F1").
    
    Parameters:
    df : pd.DataFrame
        DataFrame containing results for multiple metrics (e.g., Accuracy, F1, AUC, etc.).
    metric : str
        Metric to filter by (e.g., "Accuracy", "F1", etc.).
    
    Returns:
    pd.DataFrame : Filtered DataFrame containing only the columns for the specified metric.
    """
    # Filter columns that include the specified metric in their name
    metric_columns = [col for col in df.columns if metric in col]
    
    # Return a filtered dataframe that includes the dataset names and the filtered metric columns
    return df[['Dataset'] + metric_columns]

df_TSR_results = pd.read_excel('C:/Users/BOUKA/OneDrive/Bureau/CODE/results_averaged/TSR_averaged_results.xlsx')  # Load TSR results
df_IF_results = pd.read_excel('C:/Users/BOUKA/OneDrive/Bureau/CODE/results_averaged/IF_averaged_results.xlsx')  # Load IF results

#
# Variable: TSR
#
metrics = ['Averaged_Accuracy', 'Averaged_Balanced_accuracy', 'Averaged_F1', 'Averaged_Auc', 'Averaged_Trtime']

metrics_ = ['Trtime']
for metric in metrics_:
    results = filter_by_metric(df_TSR_results, metric=metric)
    cols_=results.columns[1:]
    print(f" the cols_: {cols_}")
    results_=results[cols_]
    scores = results_[cols_].values
    labels = ["TSR1", "TSR2", "TSR3"]
    # Plot the Critical Difference Diagram
    fig, ax = plot_critical_difference(  # Unpack the tuple (fig, ax)
        scores, 
        labels, 
        test='wilcoxon',   # Use Wilcoxon test (or 'nemenyi')
        correction='holm', # Use Holm correction
        alpha=0.05,         # Significance level
        lower_better=True, # Higher accuracy is better
        reverse=True       # Lower rank on the right side
    )
    #fig.set_size_inches(8, 6) 
    #plt.subplots_adjust(left=0.1, right=0.2, top=0.9, bottom=0.8)  # You can adjust the values
    # Save the plot before showing
    fig_name = f"TSR_{metric}_CD.png" # C:\Users\BOUKA\OneDrive\Bureau\CODE\plots
    save_path = os.path.join(save_dir, fig_name) 
    fig.savefig(save_path, bbox_inches='tight')  # Save using the figure object
    
    # Show the plot
    plt.show()

#
# Variable: IF
metrics_ = ['Trtime']
for metric in metrics_:
    results = filter_by_metric(df_IF_results, metric=metric)
    cols_=results.columns[1:]
    results_=results[cols_]
    print(f"cols of labels:{cols_}")
    scores = results_[cols_].values
    labels = ["IF1", "IF2", "IF3", "IF4", "IF5"]
    # Plot the Critical Difference Diagram
    fig, ax = plot_critical_difference(  # Unpack the tuple (fig, ax)
        scores, 
        labels, 
        test='wilcoxon',   # Use Wilcoxon test (or 'nemenyi')
        correction='holm', # Use Holm correction
        alpha=0.05,         # Significance level
        lower_better=True, # Higher accuracy is better
        reverse=True       # Lower rank on the right side
    )
    #fig.set_size_inches(8, 6) 
    #plt.subplots_adjust(left=0.1, right=0.2, top=0.9, bottom=0.8)  # You can adjust the values
    # Save the plot before showing
    fig_name = f"IF_{metric}_CD.png" # C:\Users\BOUKA\OneDrive\Bureau\CODE\plots
    save_path = os.path.join(save_dir, fig_name) 
    fig.savefig(save_path,  bbox_inches='tight')  # Save using the figure object
    
    # Show the plot
    plt.show()

from statds.no parametrics import friedman, holm
#dataset = pd.read csv(”dataset.csv”)
alpha = 0.05
columns = list(dataset.columns)
rankings, statistic, p value, critical value, hypothesis = friedman(dataset, alpha, minimize= False)
print(hypothesis)
print(”Statistic {statistic}, Rejected Value {rejected value}, p−value {p value}”)
print(rankings)
num cases = dataset.shape[0]
results, figure = holm(rankings, num cases, alpha, control = None, type rank = ”Friedman” )
print(results)
figure.show()

 
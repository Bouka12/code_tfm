import pandas as pd
import os
import scikit_posthocs as sp
import scipy.stats as ss
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
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
    metric_columns = [col for col in df.columns if metric == col]
    
    # Return a filtered dataframe that includes the dataset names and the filtered metric columns
    return df[metric_columns]

STSF_results = pd.read_excel('C:/Users/BOUKA/OneDrive/Bureau/CODE/Quant_results.xlsx')  # Load TSR results
QUANT_results = pd.read_excel('C:/Users/BOUKA/OneDrive/Bureau/CODE/STSF_results.xlsx')  # Load IF results
RSTSF_results = pd.read_excel('C:/Users/BOUKA/OneDrive/Bureau/CODE/RSTSF_results.xlsx')  # Load TSR results
TSF_results = pd.read_excel("C:/Users/BOUKA/OneDrive/Bureau/CODE/TSF_results.xlsx")
IS2_TSR2_IF4_results = pd.read_excel("C:/Users/BOUKA/OneDrive/Bureau/CODE/results_IS2/IS2_TSR2_IF4_accuracy.xlsx")
# List of file paths for the different metrics
file_paths = [
    "C:/Users/BOUKA/OneDrive/Bureau/CODE/results_IS2/IS2_TSR2_IF4_accuracy.xlsx",
    "C:/Users/BOUKA/OneDrive/Bureau/CODE/results_IS2/IS2_TSR2_IF4_balanced_accuracy.xlsx",
    "C:/Users/BOUKA/OneDrive/Bureau/CODE/results_IS2/IS2_TSR2_IF4_f1.xlsx",
    "C:/Users/BOUKA/OneDrive/Bureau/CODE/results_IS2/IS2_TSR2_IF4_auc.xlsx",
    "C:/Users/BOUKA/OneDrive/Bureau/CODE/results_IS2/IS2_TSR2_IF4_trtime.xlsx"
]

# List of metric names corresponding to the files (to rename columns later)
metric_names = ["Accuracy", "Balanced_accuracy", "F1", "Auc", "Trtime"]

# Initialize an empty list to store DataFrames
def get_results(combination):
    df_list = []

    # Loop over each file and load the data
    for metric_name in  metric_names:
        # Load the Excel file into a DataFrame
        file_path = f"C:/Users/BOUKA/OneDrive/Bureau/CODE/results_IS2/{combination}_{metric_name.lower()}.xlsx"
        df = pd.read_excel(file_path)
    
        # Rename the metric column to the metric name (assuming the metric column is the second one)
        df.columns = ['Dataset', metric_name]
    
        # Append the DataFrame to the list
        df_list.append(df)

    # Merge all DataFrames on the "Dataset" column
    final_df = df_list[0]
    for df in df_list[1:]:
        final_df = pd.merge(final_df, df, on="Dataset")
    return final_df

IS2_TSR2_IF4_results = get_results("IS2_TSR2_IF4")
IS2_TSR1_IF4_results = get_results("IS2_TSR1_IF4")
IS2_TSR3_IF4_results = get_results("IS2_TSR3_IF4")
#
# Variable: TSR
#
metrics = ["accuracy", "balanced_accuracy", "f1", "auc", "trtime"]

all_friedman_stat =[]
all_friedman_p =[]
for metric in metrics:
    RSTSF = filter_by_metric(RSTSF_results, metric=metric)
    STSF = filter_by_metric(STSF_results, metric=metric)
    Quant = filter_by_metric(QUANT_results, metric=metric)
    TSF = filter_by_metric(TSF_results, metric=metric)
    IS2_TSR2_IF4 =filter_by_metric(IS2_TSR2_IF4_results, metric = metric.capitalize())
    IS2_TSR1_IF4 = filter_by_metric(IS2_TSR1_IF4_results, metric = metric.capitalize())
    IS2_TSR3_IF4 = filter_by_metric(IS2_TSR3_IF4_results, metric = metric.capitalize())
    # Combine the results into a single DataFrame or array
    results_combined = pd.concat([RSTSF, STSF, Quant, IS2_TSR2_IF4, IS2_TSR1_IF4, IS2_TSR3_IF4, TSF], axis=1)
    print(results_combined.head())
    
    # Get the scores as a numpy array (each row corresponds to a method's results for the metric)
    scores = results_combined.values
    
    # Define the labels for the methods
    labels = ["RSTSF", "STSF", "QUANT", "IS2_TSR2_IF4", "IS2_TSR1_IF4", "IS2_TSR3_IF4", "TSF"]
    stat, p = friedmanchisquare(*scores.T)
    print(f" Friedman test on metric for all combinations: stat={all_friedman_stat}, p_values={all_friedman_p}")
    all_friedman_stat.append(stat)
    all_friedman_p.append(p)
    # Plot the Critical Difference Diagram
    """
    fig, ax = plot_critical_difference(
        scores, 
        labels, 
        test='wilcoxon',   # Use Wilcoxon test (or 'nemenyi')
        correction='holm', # Use Holm correction
        alpha=0.05,        # Significance level
        lower_better=True # Higher accuracy is better (set False for accuracy metrics)
    )
    
    # Save the plot before showing
    fig_name = f"Comparison_TSC_{metric}_CD.png"
    save_path = os.path.join(save_dir, fig_name) 
    fig.savefig(save_path, bbox_inches='tight')  # Save using the figure object
    """
    # Show the plot
    plt.show()

all_friedman_results_dict = {"metric":metrics,
                        "Friedman stat" : all_friedman_stat,
                        "p-value": all_friedman_p}
all_friedman_results = pd.DataFrame.from_dict(all_friedman_results_dict)
print(f"all models Friedman results: {all_friedman_results}")
output_path = 'C:/Users/BOUKA/OneDrive/Bureau/CODE/Comparison_SOTA_models_Friedman_results.xlsx'
all_friedman_results.to_excel(output_path, index=False)
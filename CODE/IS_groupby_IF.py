# analysis_IS_IF_grouping.py
import os
import pandas as pd

# Define the folder containing the result Excel files
folder_path = 'C:/Users/BOUKA/OneDrive/Bureau/CODE/results_test'

# Define the different combinations for IS, TSR, and IF
IS_variants = ['IS0', 'IS2']  # Interval Selection methods
metrics = ['accuracy', 'f1', 'auc', 'balanced_accuracy', 'trtime']  # Metrics
TSR_variants = ['TSR1', 'TSR2', 'TSR3']  # Time series representations
IF_variants = ['IF1', 'IF2', 'IF3', 'IF4', 'IF5']  # Interval features

# Initialize dictionaries to store the aggregated results for each combination of IS and IF
results_dict = {}

# Function to aggregate and average results for each IS and IF combination
def aggregate_results_for_IS_and_IF():
    """Aggregate results for each combination of IS and IF."""
    # Loop through IS variants
    for IS_var in IS_variants:
        results_dict[IS_var] = {}  # Initialize dictionary for each IS group

        # Loop through IF variants within each IS group
        for IF_var in IF_variants:
            combination_key = f"{IS_var}_{IF_var}"
            results_dict[IS_var][IF_var] = {metric: [] for metric in metrics}  # Initialize for each IF

            # Loop through TSR variants and aggregate the results
            for TSR in TSR_variants:
                combination = f"{IS_var}_{TSR}_{IF_var}"
                
                # For each metric, read the corresponding Excel file and append to results
                for metric in metrics:
                    file_path = os.path.join(folder_path, f"{combination}_{metric}.xlsx")
                    df = pd.read_excel(file_path)  # Load the Excel file
                    capitalized_metric = metric.capitalize()

                    # Append the metric values from the file to the corresponding IS_IF group
                    results_dict[IS_var][IF_var][metric].append(df[capitalized_metric])

            # Once all TSR variants for this IF are processed, average across them
            for metric in metrics:
                combined_df = pd.concat(results_dict[IS_var][IF_var][metric], axis=1)
                results_dict[IS_var][IF_var][metric] = combined_df.mean(axis=1)  # Average across TSR variants

# Process results for all IS and IF combinations
aggregate_results_for_IS_and_IF()

# Create a final DataFrame for storing results grouped by IS and IF variants
final_df_IS_IF = pd.DataFrame()

# Assuming all datasets are the same across metrics, get the dataset names from one of the files
first_combination = f"{IS_variants[0]}_{TSR_variants[0]}_{IF_variants[0]}"
dataset_file = os.path.join(folder_path, f"{first_combination}_accuracy.xlsx")
df_datasets = pd.read_excel(dataset_file)

# Add the dataset names as the first column (assuming 'Dataset' column exists in all files)
final_df_IS_IF['Dataset'] = df_datasets['Dataset']

# Add the results for each IS and IF combination
for IS_var in IS_variants:
    for IF_var in IF_variants:
        for metric in metrics:
            column_name = f"{IS_var}_{IF_var}_Averaged_{metric.capitalize()}"
            final_df_IS_IF[column_name] = results_dict[IS_var][IF_var][metric]
'''
# Save the final results to an Excel file
output_file_path = 'C:/Users/BOUKA/OneDrive/Bureau/CODE/results_averaged/Results_Grouped_by_IS_and_IF.xlsx'
final_df_IS_IF.to_excel(output_file_path, index=False)

print("Results saved to Excel file successfully!")
'''
import pandas as pd
from scipy.stats import friedmanchisquare
df = final_df_IS_IF.copy()

# Assuming 'df' is your DataFrame
#metrics = ['Averaged_Accuracy', 'Averaged_F1', 'Averaged_Auc', 'Averaged_Balanced_accuracy', 'Averaged_Trtime']
combinations = ['IS0_IF1', 'IS0_IF2', 'IS0_IF3', 'IS0_IF4', 'IS0_IF5', 
                'IS2_IF1', 'IS2_IF2', 'IS2_IF3', 'IS2_IF4', 'IS2_IF5']
'''
# Create a dictionary to store the results of the Friedman test
friedman_results = {'Metric': [], 'Friedman_stat': [], 'p-value': []}

# Perform the Friedman test for each metric
for metric in metrics:
    metric_columns = [f"{comb}_{metric}" for comb in combinations]  # Get columns for this metric
    data = [df[col] for col in metric_columns]  # Get the data for these columns
    
    stat, p_value = friedmanchisquare(*data)
    friedman_results['Metric'].append(metric)
    friedman_results['Friedman_stat'].append(stat)
    friedman_results['p-value'].append(p_value)

# Convert to DataFrame and save the results to an Excel file
friedman_df = pd.DataFrame(friedman_results)
friedman_df.to_excel(f'C:/Users/BOUKA/OneDrive/Bureau/CODE/results_averaged/IS-IF_friedman_test_results.xlsx', index=False)

print("Friedman test results saved to Excel.")
'''
save_dir ="C:/Users/BOUKA/OneDrive/Bureau/CODE/results_averaged/"
from scipy.stats import wilcoxon
import itertools
'''
# Create a dictionary to store the Wilcoxon test results
wilcoxon_results = {'Comparison': [], 'Metric': [], 'Wilcoxon_stat': [], 'p-value': []}

# Generate all pairwise combinations of IS-IF for each metric
for metric in metrics:
    metric_columns = [f"{comb}_Averaged_{metric.capitalize()}" for comb in combinations]
    for comb1, comb2 in itertools.combinations(metric_columns, 2):
        stat, p_value = wilcoxon(df[comb1], df[comb2])
        comparison = f'{comb1} vs {comb2}'
        wilcoxon_results['Comparison'].append(comparison)
        wilcoxon_results['Metric'].append(metric)
        wilcoxon_results['Wilcoxon_stat'].append(stat)
        wilcoxon_results['p-value'].append(p_value)

# Convert to DataFrame and save results to an Excel file
wilcoxon_df = pd.DataFrame(wilcoxon_results)
wilcoxon_df.to_excel(f'{save_dir}IS-IF_wilcoxon_test_results.xlsx', index=False)

print("Wilcoxon test results saved to Excel.")
'''
'''
# Define p-value categories
pvalue_categories = {
    '< 0.001': 0,
    '0.001 - 0.05': 0,
    '> 0.05': 0
}

# Create a dictionary to store the Wilcoxon test results and the categorized p-values
wilcoxon_results = {'Comparison': [], 'Metric': [], 'Wilcoxon_stat': [], 'p-value': []}
pvalue_count_by_metric = {metric: pvalue_categories.copy() for metric in metrics}

# Generate all pairwise combinations of IS-IF for each metric
for metric in metrics:
    metric_columns = [f"{comb}_Averaged_{metric.capitalize()}" for comb in combinations]
    for comb1, comb2 in itertools.combinations(metric_columns, 2):
        stat, p_value = wilcoxon(df[comb1], df[comb2])
        comparison = f'{comb1} vs {comb2}'
        wilcoxon_results['Comparison'].append(comparison)
        wilcoxon_results['Metric'].append(metric)
        wilcoxon_results['Wilcoxon_stat'].append(stat)
        wilcoxon_results['p-value'].append(p_value)
        
        # Categorize the p-value
        if p_value < 0.001:
            pvalue_count_by_metric[metric]['< 0.001'] += 1
        elif 0.001 <= p_value <= 0.05:
            pvalue_count_by_metric[metric]['0.001 - 0.05'] += 1
        else:
            pvalue_count_by_metric[metric]['> 0.05'] += 1

# Convert to DataFrame and save Wilcoxon test results to an Excel file
wilcoxon_df = pd.DataFrame(wilcoxon_results)
wilcoxon_df.to_excel('wilcoxon_test_results.xlsx', index=False)

# Convert p-value counts to DataFrame
pvalue_count_df = pd.DataFrame.from_dict(pvalue_count_by_metric, orient='index')
pvalue_count_df.to_excel(f'{save_dir}IS-IF_pvalue_categorization_results.xlsx', index=True)

print("Wilcoxon test results and p-value categorization saved to Excel.")
'''
import os
import matplotlib.pyplot as plt

# Assuming df_IF_results is your DataFrame containing the metrics for each IF combination
# and plot_critical_difference is a predefined function for generating the CD plot

metrics = ['Accuracy', 'F1', 'Auc', 'Balanced_accuracy', 'Trtime']  # List of metrics
save_dir = "C:/Users/BOUKA/OneDrive/Bureau/CODE/plots"  # Directory to save plots

# Function to filter results by metric
def filter_by_metric(df, metric):
    return df.filter(like=f"Averaged_{metric}")

from aeon.visualisation import plot_critical_difference

metrics_=['Trtime']
# Loop through each metric and generate the CD plot
for metric in metrics_:
    # Filter the DataFrame for the current metric
    results = filter_by_metric(df, metric=metric)
    cols_ = results.columns  # Assuming the first column is "Dataset", so skip it
    results_ = results[cols_]
    
    print(f"Columns for labels: {cols_}")
    
    # Extract the scores and labels (assuming the labels correspond to IF1, IF2, etc.)
    scores = results_.values  # Get the values for each IF combination
    labels = ["_".join(col.split('_')[:2]) for col in cols_] # Extract labels like IF1, IF2, etc.
    
    # Plot the Critical Difference Diagram
    fig, ax = plot_critical_difference(
        scores=scores, 
        labels=labels, 
        test='wilcoxon',   # Using Wilcoxon test for pairwise comparisons
        correction='holm', # Apply Holm correction for multiple comparisons
        alpha=0.05,        # Significance level of 5%
        lower_better=True, # Higher values are better (e.g., higher accuracy is better)
        reverse= True      # Do not reverse the ranks
    )
    fig.set_size_inches(9, 7) 
    #plt.subplots_adjust(left=0.1, right=0.2, top=0.9, bottom=0.8)  # You can adjust the values
    # Define the filename for the plot
    fig_name = f"IS-IF_{metric}_CD.png"
    save_path = os.path.join(save_dir, fig_name)
    
    # Save the figure
    fig.savefig(save_path, bbox_inches="tight")  # Save the figure with tight bounding box
    
    # Show the plot
    plt.show()

print("Critical Difference plots generated and saved.")

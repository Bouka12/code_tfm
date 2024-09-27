# plots.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as snss
from scipy.stats import wilcoxon
import os

# Define the combinations for IS, TSR, IF
IS_options = ['IS0', 'IS2']
TSR_options = ['TSR1', 'TSR2', 'TSR3']
IF_options = ['IF1', 'IF2', 'IF3', 'IF4', 'IF5']
metrics = ['Accuracy', 'F1', 'Auc', 'Balanced_accuracy', 'Trtime']

# Initialize a dictionary to store the data for each metric
all_results = {metric: pd.DataFrame() for metric in metrics}
dir = "C:/Users/BOUKA/OneDrive/Bureau/CODE/results_test/"

# Load all the data for each combination of IS, TSR, and IF
combinations = []
for IS in IS_options:
    for TSR in TSR_options:
        for IF in IF_options:
            combination_name = f"{IS}_{TSR}_{IF}"
            combinations.append(combination_name)
            for metric in metrics:
                # Assume each combination has a corresponding file for each metric
                file_name = f"{combination_name}_{metric}.xlsx"
                file_dir = os.path.join(dir, file_name)
                if os.path.exists(file_dir):
                    df = pd.read_excel(file_dir)
                    df.set_index('Dataset', inplace=True)  # Assume the 'Dataset' column is the index
                    all_results[metric][combination_name] = df[metric]

# Perform Wilcoxon test for all models (combinations) across each metric
wilcoxon_results = {metric: pd.DataFrame(index=combinations, columns=combinations) for metric in metrics}
pvalue_categories = {
    '< 0.001': 0,
    '0.001 - 0.05': 0,
    '> 0.05': 0
}
# Loop through each metric and perform pairwise Wilcoxon tests between model combinations
pvalue_count_by_metric = {metric: pvalue_categories.copy() for metric in metrics}

for metric in metrics:
    for i, model1 in enumerate(combinations):
        for j, model2 in enumerate(combinations):
            if i < j:  # Perform the test for unique pairs
                stat, p_value = wilcoxon(all_results[metric][model1], all_results[metric][model2])
                wilcoxon_results[metric].loc[model1, model2] = p_value
                wilcoxon_results[metric].loc[model2, model1] = p_value  # Fill the symmetric part of the matrix
                # Categorize the p-value
                if p_value < 0.001:
                    pvalue_count_by_metric[metric]['< 0.001'] += 1
                elif 0.001 <= p_value <= 0.05:
                    pvalue_count_by_metric[metric]['0.001 - 0.05'] += 1
                else:
                    pvalue_count_by_metric[metric]['> 0.05'] += 1
# Save the Wilcoxon test results for each metric into an Excel file
'''
output_path = 'C:/Users/BOUKA/OneDrive/Bureau/CODE/results_averaged/All_models_Wilcoxon_results.xlsx'
with pd.ExcelWriter(output_path) as writer:
    for metric, df in wilcoxon_results.items():
        df.to_excel(writer, sheet_name=f'{metric}')



print("Wilcoxon test results and p-value categorization saved to Excel.")
'''

# Define the combinations for IS, TSR, IF
IS_options = ['IS0', 'IS2']
TSR_options = ['TSR1', 'TSR2', 'TSR3']
IF_options = ['IF1', 'IF2', 'IF3', 'IF4', 'IF5']
metrics = ['Accuracy', 'F1', 'Auc', 'Balanced_accuracy', 'Trtime']

# Initialize a dictionary to store the data for each metric
all_results = {metric: pd.DataFrame() for metric in metrics}
dir = "C:/Users/BOUKA/OneDrive/Bureau/CODE/results_test/"

# Loop through each combination of IS, TSR, IF
for IS in IS_options:
    for TSR in TSR_options:
        for IF in IF_options:
            combination_name = f"{IS}_{TSR}_{IF}"
            for metric in metrics:
                # Assume each combination has a corresponding file for each metric
                file_name = f"{combination_name}_{metric}.xlsx"  # Adjust this if the file names are different
                file_dir = os.path.join(dir, file_name)
                if os.path.exists(file_dir):
                    df = pd.read_excel(file_dir)  # Load the Excel file
                    df.set_index('Dataset', inplace=True)  # Assume the 'Dataset' column is the index
                    # Add this combination's results to the corresponding metric DataFrame
                    all_results[metric][combination_name] = df[metric]  # Replace 'value' with the actual column name
#print(all_results)
#output_path = 'C:/Users/BOUKA/OneDrive/Bureau/CODE/results_averaged/all_results.xlsx'
#all_results.to_excel(output_path, index=False)

from aeon.visualisation import plot_critical_difference
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare
# Loop through each metric to generate and save the plots
'''
all_friedman_p = []
all_friedman_stat=[]
for metric in metrics:
    # Get the scores for all 30 combinations across 112 datasets
    scores = all_results[metric].values  # This should be (112, 30)
    labels = all_results[metric].columns  # Combination names (e.g., IS0_TSR1_IF1, etc.)
    
    stat, p = friedmanchisquare(*scores.T)
    print(f" Friedman test on metric for all combinations: stat={all_friedman_stat}, p_values={all_friedman_p}")
    all_friedman_stat.append(stat)
    all_friedman_p.append(p)
     
    # Plot the Critical Difference Diagram
    fig, ax = plot_critical_difference(
        scores, 
        labels, 
        test='wilcoxon',   # Use Wilcoxon test (or 'nemenyi')
        correction='holm', # Use Holm correction
        alpha=0.05,         # Significance level
        lower_better=False, # Higher scores are better
        reverse=True       # Lower rank on the right side
    )
    fig.set_size_inches(12, 8) 
    plt.subplots_adjust(left=0.1, right=0.2, top=0.9, bottom=0.8)  # You can adjust the values

        # Adjust subplots to add more space around the plot
    #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Change values as needed

    # Show the plot
    fig_name = f"All_CD_Plot_{metric}.png"  # Adjust this if the file names are different
    save_dir = os.path.join("C:/Users/BOUKA/OneDrive/Bureau/CODE/plots/", fig_name)
    # Optionally save the plot
    fig.savefig(save_dir,  bbox_inches='tight')
    plt.show()

''' 


'''
all_friedman_results_dict = {"metric":metrics,
                        "Friedman stat" : all_friedman_stat,
                        "p-value": all_friedman_p}
all_friedman_results = pd.DataFrame.from_dict(all_friedman_results_dict)
print(f"all models Friedman results: {all_friedman_results}")
output_path = 'C:/Users/BOUKA/OneDrive/Bureau/CODE/results_averaged/All_models_Friedman_results.xlsx'
all_friedman_results.to_excel(output_path, index=False)


# Wilcoxon test on each group of datasets
'''
#metrics_=['Trtime']
metrics = ['Accuracy', 'F1', 'Auc', 'Balanced_accuracy']
for metric in metrics:
    # Get the scores for all 30 combinations across 112 datasets
    scores = all_results[metric].values  # This should be (112, 30)
    labels = all_results[metric].columns  # Combination names (e.g., IS0_TSR1_IF1, etc.)
    '''
    stat, p = friedmanchisquare(*scores.T)
    print(f" Friedman test on metric for all combinations: stat={all_friedman_stat}, p_values={all_friedman_p}")
    all_friedman_stat.append(stat)
    all_friedman_p.append(p)
    ''' 
    # Plot the Critical Difference Diagram
    fig, ax = plot_critical_difference(
        scores, 
        labels, 
        test='wilcoxon',   # Use Wilcoxon test (or 'nemenyi')
        correction='holm', # Use Holm correction
        alpha=0.05,         # Significance level
        lower_better=False, # Higher scores are better
        reverse=True       # Lower rank on the right side
    )
    fig.set_size_inches(12, 8) 
    plt.subplots_adjust(left=0.1, right=0.2, top=0.9, bottom=0.8)  # You can adjust the values

        # Adjust subplots to add more space around the plot
    #plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)  # Change values as needed

    # Show the plot
    fig_name = f"All_CD_Plot_{metric}.png"  # Adjust this if the file names are different
    save_dir = os.path.join("C:/Users/BOUKA/OneDrive/Bureau/CODE/plots/", fig_name)
    # Optionally save the plot
    fig.savefig(save_dir,  bbox_inches='tight')
    plt.show()

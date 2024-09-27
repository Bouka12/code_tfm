# analysis_IS.py
import os
import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare

# Define the folder containing the result Excel files
folder_path = 'C:/Users/BOUKA/OneDrive/Bureau/CODE/results_test'

# Define the different combinations for IS, TSR, and IF
IS_variants = ['IS0', 'IS2']  # Interval Selection methods
metrics = ['accuracy', 'f1', 'auc', 'balanced_accuracy', 'trtime']  # Metrics
TSR_variants = ['TSR1', 'TSR2', 'TSR3']  # Time series representations
IF_variants = ['IF1', 'IF2', 'IF3', 'IF4', 'IF5']  # Interval features

# Initialize dictionaries to store the aggregated results
results_dict = {'IS0': {}, 'IS2': {}}

# Function to aggregate and average results
def aggregate_results_for_IS(IS_var):
    """Aggregate results for a given interval selection method."""
    combined_results = {metric: [] for metric in metrics}

    # Loop through TSR and IF variants for the given IS variant
    for TSR in TSR_variants:
        for IF in IF_variants:
            combination = f"{IS_var}_{TSR}_{IF}"
            
            # For each metric, read the corresponding Excel file and append to combined results
            for metric in metrics:
                file_path = os.path.join(folder_path, f"{combination}_{metric}.xlsx")
                df = pd.read_excel(file_path)  # Load the Excel file
                capitalized_metric = metric.capitalize()
                #print(f"The metric capitalized: {capitalized_metric}")
                combined_results[metric].append(df[capitalized_metric])  # Append the metric values from the file

    # For each metric, concatenate the results and compute the average across all datasets
    for metric in metrics:
        combined_df = pd.concat(combined_results[metric], axis=1)
        results_dict[IS_var][metric] = combined_df.mean(axis=1)  # Average across all combinations

# Process results for both IS0 and IS2
for IS_var in IS_variants:
    aggregate_results_for_IS(IS_var)
'''
# Display the averaged results
for IS_var in IS_variants:
    print(f"Results for {IS_var}:")
    for metric in metrics:
        print(f"\nAveraged {metric.capitalize()}:\n")
        print(results_dict[IS_var][metric])
'''
# Combine the averaged results into a single DataFrame
final_df_IS = pd.DataFrame()

# Assuming all datasets are the same across metrics and IS variants, get the dataset names from one of the files
first_combination = f"{IS_variants[0]}_{TSR_variants[0]}_{IF_variants[0]}"
dataset_file = os.path.join(folder_path, f"{first_combination}_accuracy.xlsx")
df_datasets = pd.read_excel(dataset_file)

# Add the dataset names as the first column
final_df_IS['Dataset'] = df_datasets['Dataset']  # Assuming the first column is 'Dataset' in all Excel files

# Add the results for each metric and IS variant
for IS_var in IS_variants:
    for metric in metrics:
        column_name = f"{IS_var}_Averaged_{metric.capitalize()}"
        final_df_IS[column_name] = results_dict[IS_var][metric]

# Save the combined DataFrame to an Excel file
#output_path = 'C:/Users/BOUKA/OneDrive/Bureau/CODE/IS_averaged_results.xlsx'
#final_df_IS.to_excel(output_path, index=False)

# Print the final DataFrame
#print(final_df_IS)
# Initialize dictionaries to store the aggregated results
results_dict_TSR = {tsr: {} for tsr in TSR_variants}
results_dict_IF = {if_var: {} for if_var in IF_variants}

# Function to aggregate and average results for TSR
def aggregate_results_for_TSR(TSR_var):
    """Aggregate results for a given Time Series Representation (TSR) method."""
    combined_results = {metric: [] for metric in metrics}

    # Loop through IS and IF variants for the given TSR variant
    for IS in IS_variants:
        for IF in IF_variants:
            combination = f"{IS}_{TSR_var}_{IF}"
            
            # For each metric, read the corresponding Excel file and append to combined results
            for metric in metrics:
                file_path = os.path.join(folder_path, f"{combination}_{metric}.xlsx")
                df = pd.read_excel(file_path)  # Load the Excel file
                capitalized_metric = metric.capitalize()
                combined_results[metric].append(df[capitalized_metric])  # Append the metric values from the file

    # For each metric, concatenate the results and compute the average across all datasets
    for metric in metrics:
        combined_df = pd.concat(combined_results[metric], axis=1)
        results_dict_TSR[TSR_var][metric] = combined_df.mean(axis=1)  # Average across all combinations

# Function to aggregate and average results for IF
def aggregate_results_for_IF(IF_var):
    """Aggregate results for a given Interval Features (IF) method."""
    combined_results = {metric: [] for metric in metrics}

    # Loop through IS and TSR variants for the given IF variant
    for IS in IS_variants:
        for TSR in TSR_variants:
            combination = f"{IS}_{TSR}_{IF_var}"
            
            # For each metric, read the corresponding Excel file and append to combined results
            for metric in metrics:
                file_path = os.path.join(folder_path, f"{combination}_{metric}.xlsx")
                df = pd.read_excel(file_path)  # Load the Excel file
                capitalized_metric = metric.capitalize()
                combined_results[metric].append(df[capitalized_metric])  # Append the metric values from the file

    # For each metric, concatenate the results and compute the average across all datasets
    for metric in metrics:
        combined_df = pd.concat(combined_results[metric], axis=1)
        results_dict_IF[IF_var][metric] = combined_df.mean(axis=1)  # Average across all combinations

# Process results for TSR
for TSR_var in TSR_variants:
    aggregate_results_for_TSR(TSR_var)

# Process results for IF
for IF_var in IF_variants:
    aggregate_results_for_IF(IF_var)

# Combine the averaged results into separate DataFrames for TSR and IF
final_df_TSR = pd.DataFrame()
final_df_IF = pd.DataFrame()

# Assuming all datasets are the same across metrics and TSR/IF variants, get the dataset names from one of the files
first_combination_TSR = f"{IS_variants[0]}_{TSR_variants[0]}_{IF_variants[0]}"
first_combination_IF = f"{IS_variants[0]}_{TSR_variants[0]}_{IF_variants[0]}"
dataset_file = os.path.join(folder_path, f"{first_combination_TSR}_accuracy.xlsx")
df_datasets = pd.read_excel(dataset_file)

# Add the dataset names as the first column in both DataFrames
final_df_TSR['Dataset'] = df_datasets['Dataset']  # Assuming the first column is 'Dataset' in all Excel files
final_df_IF['Dataset'] = df_datasets['Dataset']  # Assuming the first column is 'Dataset' in all Excel files

# Add the results for each metric and TSR/IF variant
for TSR_var in TSR_variants:
    for metric in metrics:
        column_name = f"{TSR_var}_Averaged_{metric.capitalize()}"
        final_df_TSR[column_name] = results_dict_TSR[TSR_var][metric]

for IF_var in IF_variants:
    for metric in metrics:
        column_name = f"{IF_var}_Averaged_{metric.capitalize()}"
        final_df_IF[column_name] = results_dict_IF[IF_var][metric]

# Save the combined DataFrames to Excel files
#output_path_TSR = 'C:/Users/BOUKA/OneDrive/Bureau/CODE/TSR_averaged_results.xlsx'
#output_path_IF = 'C:/Users/BOUKA/OneDrive/Bureau/CODE/IF_averaged_results.xlsx'
#final_df_TSR.to_excel(output_path_TSR, index=False)
#final_df_IF.to_excel(output_path_IF, index=False)

# Print the final DataFrames
#print("TSR Averaged Results:")
#print(final_df_TSR)
#print("\nIF Averaged Results:")
#print(final_df_IF)

'''
# Friedmann test for difference in IF variables and TSR variables:
# Accuracy:
TSR_accuracy_stat, TSR_accuracy_p = friedmanchisquare(final_df_TSR['TSR1_Averaged_Accuracy'], final_df_TSR['TSR2_Averaged_Accuracy'], final_df_TSR['TSR3_Averaged_Accuracy'])
print(f'Friedman test for Accuracy in TSR: statistic={TSR_accuracy_stat}, p-value={TSR_accuracy_p}')
# Balanced accuracy
TSR_Balanced_accuracy_stat, TSR_Balanced_accuracy_p = friedmanchisquare(final_df_TSR['TSR1_Averaged_Balanced_accuracy'], final_df_TSR['TSR2_Averaged_Balanced_accuracy'], final_df_TSR['TSR3_Averaged_Balanced_accuracy'])
print(f'Friedman test for Balanced accuracy in TSR: statistic={TSR_Balanced_accuracy_stat}, p-value={TSR_Balanced_accuracy_p}')
# AUC
TSR_auc_stat, TSR_auc_p = friedmanchisquare(final_df_TSR['TSR1_Averaged_Auc'], final_df_TSR['TSR2_Averaged_Auc'], final_df_TSR['TSR3_Averaged_Auc'])
print(f'Friedman test for AUC in TSR: statistic={TSR_auc_stat}, p-value={TSR_auc_p}')
# F1
TSR_f1_stat, TSR_f1_p = friedmanchisquare(final_df_TSR['TSR1_Averaged_F1'], final_df_TSR['TSR2_Averaged_F1'], final_df_TSR['TSR3_Averaged_F1'])
print(f'Friedman test for f1 in TSR: statistic={TSR_f1_stat}, p-value={TSR_f1_p}')
# Trtime
TSR_trtime_stat, TSR_trtime_p = friedmanchisquare(final_df_TSR['TSR1_Averaged_Trtime'], final_df_TSR['TSR2_Averaged_Trtime'], final_df_TSR['TSR3_Averaged_Trtime'])
print(f'Friedman test for trtime in TSR: statistic={TSR_trtime_stat}, p-value={TSR_trtime_p}')
friedman_tsr_stats = [TSR_accuracy_stat, TSR_Balanced_accuracy_stat, TSR_auc_stat, TSR_f1_stat,  TSR_trtime_stat]
friedman_tsr_p = [TSR_accuracy_p, TSR_Balanced_accuracy_p, TSR_auc_p,  TSR_f1_p, TSR_trtime_p ]
metric = ['Averaged_Accuracy', 'Averaged_Balanced_accuracy', 'Averaged_Auc', 'Averaged_F1','Averaged_Trtime']
Friedman_TSR_results_dict = {'Metric':metric,
                             'Friedman stat': friedman_tsr_stats,
                             'p-value':friedman_tsr_p}
Friedman_TSR_results = pd.DataFrame.from_dict(Friedman_TSR_results_dict)
print(f"Friedman resutls for the TSR componenet: {Friedman_TSR_results}")
# Save results as EXCEL file
output_path = 'C:/Users/BOUKA/OneDrive/Bureau/CODE/results_averaged/TSR_Friedman_results.xlsx'
Friedman_TSR_results.to_excel(output_path, index=False)
'''
# Friedman test to determine if there is a significant difference in the variables of the IF
'''
# accuracy
IF_accuracy_stat, IF_accuracy_p = friedmanchisquare(final_df_IF['IF1_Averaged_Accuracy'], final_df_IF['IF2_Averaged_Accuracy'], final_df_IF['IF3_Averaged_Accuracy'], final_df_IF['IF4_Averaged_Accuracy'], final_df_IF['IF5_Averaged_Accuracy'])
print(f'Friedman test for Accuracy in IF: statistic={IF_accuracy_stat}, p-value={IF_accuracy_p}')

# balanced accuracy

IF_Balanced_accuracy_stat, IF_Balanced_accuracy_p = friedmanchisquare(final_df_IF['IF1_Averaged_Balanced_accuracy'], final_df_IF['IF2_Averaged_Balanced_accuracy'], final_df_IF['IF3_Averaged_Balanced_accuracy'], final_df_IF['IF4_Averaged_Balanced_accuracy'], final_df_IF['IF5_Averaged_Balanced_accuracy'])
print(f'Friedman test for Balanced accuracy in IF: statistic={IF_Balanced_accuracy_stat}, p-value={IF_Balanced_accuracy_p}')

# AUC

IF_auc_stat, IF_auc_p = friedmanchisquare(final_df_IF['IF1_Averaged_Auc'], final_df_IF['IF2_Averaged_Auc'], final_df_IF['IF3_Averaged_Auc'], final_df_IF['IF4_Averaged_Auc'], final_df_IF['IF5_Averaged_Auc'])
print(f'Friedman test for AUC in IF: statistic={IF_auc_stat}, p-value={IF_auc_p}')

# F1
IF_f1_stat, IF_f1_p = friedmanchisquare(final_df_IF['IF1_Averaged_F1'], final_df_IF['IF2_Averaged_F1'], final_df_IF['IF3_Averaged_F1'], final_df_IF['IF4_Averaged_F1'], final_df_IF['IF5_Averaged_F1'])
print(f'Friedman test for F1 in IF: statistic={IF_f1_stat}, p-value={IF_f1_p}')

# Trtime
IF_trtime_stat, IF_trtime_p = friedmanchisquare(final_df_IF['IF1_Averaged_Trtime'], final_df_IF['IF2_Averaged_Trtime'], final_df_IF['IF3_Averaged_Trtime'], final_df_IF['IF4_Averaged_Trtime'], final_df_IF['IF5_Averaged_Trtime'])
print(f'Friedman test for trtime in IF: statistic={IF_trtime_stat}, p-value={IF_trtime_p}')

# Save the results
friedman_IF_stats = [IF_accuracy_stat, IF_Balanced_accuracy_stat, IF_auc_stat, IF_f1_stat,  IF_trtime_stat]
friedman_IF_p = [IF_accuracy_p, IF_Balanced_accuracy_p, IF_auc_p,  IF_f1_p, IF_trtime_p ]
metric = ['Averaged_Accuracy', 'Averaged_Balanced_accuracy', 'Averaged_Auc', 'Averaged_F1','Averaged_Trtime']
Friedman_IF_results_dict = {'Metric':metric,
                             'Friedman stat': friedman_IF_stats,
                             'p-value':friedman_IF_p}
Friedman_IF_results = pd.DataFrame.from_dict(Friedman_IF_results_dict)
print(f"Friedman resutls for the IF componenet: {Friedman_IF_results}")
# Save results as EXCEL file
output_path = 'C:/Users/BOUKA/OneDrive/Bureau/CODE/results_averaged/IF_Friedman_results.xlsx'
Friedman_IF_results.to_excel(output_path, index=False)
# Analysis of the Interval selection : wilcoxon paired test
# the name of the dataset of the IS results: `final_df_IS`
'''
from scipy.stats import wilcoxon
import pandas as pd
import os

# Variables
TSR_options = ['TSR1', 'TSR2', 'TSR3']
IF_options = ['IF1', 'IF2', 'IF3', 'IF4', 'IF5']
metrics = ['Averaged_Accuracy', 'Averaged_Balanced_accuracy', 'Averaged_Auc', 'Averaged_F1', 'Averaged_Trtime']

# DataFrames to store Wilcoxon results for TSR and IF variables
wilcoxon_results_TSR = {metric: pd.DataFrame(index=TSR_options, columns=TSR_options) for metric in metrics}
wilcoxon_results_IF = {metric: pd.DataFrame(index=IF_options, columns=IF_options) for metric in metrics}

# Perform the Wilcoxon test for TSR
for metric in metrics:
    for i, TSR1 in enumerate(TSR_options):
        for j, TSR2 in enumerate(TSR_options):
            if i < j:  # Perform the test for unique pairs
                stat, p_value = wilcoxon(final_df_TSR[f'{TSR1}_{metric}'], final_df_TSR[f'{TSR2}_{metric}'])
                wilcoxon_results_TSR[metric].loc[TSR1, TSR2] = p_value
                wilcoxon_results_TSR[metric].loc[TSR2, TSR1] = p_value  # Fill the symmetric part of the matrix

# Perform the Wilcoxon test for IF
for metric in metrics:
    for i, IF1 in enumerate(IF_options):
        for j, IF2 in enumerate(IF_options):
            if i < j:  # Perform the test for unique pairs
                stat, p_value = wilcoxon(final_df_IF[f'{IF1}_{metric}'], final_df_IF[f'{IF2}_{metric}'])
                wilcoxon_results_IF[metric].loc[IF1, IF2] = p_value
                wilcoxon_results_IF[metric].loc[IF2, IF1] = p_value  # Fill the symmetric part of the matrix

# Save the results for TSR to Excel
output_path_tsr = 'C:/Users/BOUKA/OneDrive/Bureau/CODE/results_averaged/TSR_Wilcoxon_results.xlsx'
with pd.ExcelWriter(output_path_tsr) as writer:
    for metric, df in wilcoxon_results_TSR.items():
        df.to_excel(writer, sheet_name=f'{metric}')

# Save the results for IF to Excel
output_path_if = 'C:/Users/BOUKA/OneDrive/Bureau/CODE/results_averaged/IF_Wilcoxon_results.xlsx'
with pd.ExcelWriter(output_path_if) as writer:
    for metric, df in wilcoxon_results_IF.items():
        df.to_excel(writer, sheet_name=f'{metric}')

print("Wilcoxon test results saved successfully!")

'''
accuracy_stat, accuracy_p = wilcoxon(final_df_IS['IS0_Averaged_Accuracy'], final_df_IS['IS2_Averaged_Accuracy'])
Balanced_accuracy_stat, Balanced_accuracy_p = wilcoxon(final_df_IS['IS0_Averaged_Balanced_accuracy'], final_df_IS['IS2_Averaged_Balanced_accuracy'])
f1_stat, f1_p = wilcoxon(final_df_IS['IS0_Averaged_F1'], final_df_IS['IS2_Averaged_F1'])
auc_stat, auc_p = wilcoxon(final_df_IS['IS0_Averaged_Auc'], final_df_IS['IS2_Averaged_Auc'])
trtime_stat, trtime_p = wilcoxon(final_df_IS['IS0_Averaged_Trtime'], final_df_IS['IS2_Averaged_Trtime'])

# Output results
print(f'Wilcoxon test for Accuracy: statistic={accuracy_stat}, p-value={accuracy_p}')
print(f'Wilcoxon test for F1: statistic={f1_stat}, p-value={f1_p}')
print(f'Wilcoxon test for AUC: statistic={auc_stat}, p-value={auc_p}')
print(f'Wilcoxon test for Balanced accuracy: statistic={Balanced_accuracy_stat}, p-value={Balanced_accuracy_p}')
print(f'Wilcoxon test for trtime: statistic={trtime_stat}, p-value={trtime_p}')
'''
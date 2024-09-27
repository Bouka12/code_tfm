from data_loader import load_datasets
from tsc_datasets import univariate_equal_length
from aeon.visualisation import plot_critical_difference
import matplotlib.pyplot as plt
'''
data_dict = load_datasets(univariate_equal_length)
large_data_length= []
small_data_length = []
large_data_instances = []
small_data_instances =[]
for dataset in data_dict.keys():
    print(f"train shape: {data_dict[dataset]['X_train'].shape}")
    train_instances, train_length = data_dict[dataset]['X_train'].shape[0], data_dict[dataset]['X_train'].shape[2]
    test_instances, test_length = data_dict[dataset]['X_test'].shape[0], data_dict[dataset]['X_test'].shape[2]
    n_instances = train_instances +test_instances
    if n_instances>100:
        large_data_length.append(dataset)
        print(f"{dataset} is in large_data_length")
    else :
        print(f"{dataset} is in small_data_length")
        small_data_length.append(dataset)
    if n_instances>1000:
        large_data_instances.append(dataset)
        print(f"{dataset} is in large_data_instances")
    else:
        small_data_instances.append(dataset)
        print(f"{dataset} is in small_data_instances")

print(f"Large data-Instances-: {large_data_instances}")
print(f"small data-Instances-: {small_data_instances}")

print(f"large data -length-:{large_data_length}")
print(f"small data-length-:{small_data_length}")
'''
from scipy.stats import wilcoxon
import pandas as pd
import os
from tsc_datasets import large_data_instances, small_data_instances
# Define the combinations for IS, TSR, IF
IS_options = ['IS0', 'IS2']
TSR_options = ['TSR1', 'TSR2', 'TSR3']
IF_options = ['IF1', 'IF2', 'IF3', 'IF4', 'IF5']
metrics = ['Accuracy', 'F1', 'Auc', 'Balanced_accuracy', 'Trtime']

# Lists of dataset names
large_datasets_list = large_data_instances  
small_datasets_list = small_data_instances 

# Initialize a dictionary to store the data for each metric for both large and small datasets
all_results_large = {metric: pd.DataFrame() for metric in metrics}
all_results_small = {metric: pd.DataFrame() for metric in metrics}
dir = "C:/Users/BOUKA/OneDrive/Bureau/CODE/results_test/"

# Load all the data for each combination of IS, TSR, and IF and split into large and small datasets
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
                    # Split data into large and small datasets
                    all_results_large[metric][combination_name] = df.loc[large_datasets_list, metric]
                    all_results_small[metric][combination_name] = df.loc[small_datasets_list, metric]

# Perform Wilcoxon test for all models (combinations) on both large and small datasets
wilcoxon_results_large = {metric: pd.DataFrame(index=combinations, columns=combinations) for metric in metrics}
wilcoxon_results_small = {metric: pd.DataFrame(index=combinations, columns=combinations) for metric in metrics}
pvalue_categories = {
    '< 0.001': 0,
    '0.001 - 0.05': 0,
    '> 0.05': 0
}
# Loop through each metric and perform pairwise Wilcoxon tests between model combinations
pvalue_count_by_metric_large = {metric: pvalue_categories.copy() for metric in metrics}
# Wilcoxon test for large datasets
for metric in metrics:
    for i, model1 in enumerate(combinations):
        for j, model2 in enumerate(combinations):
            if i < j:  # Perform the test for unique pairs
                stat, p_value = wilcoxon(all_results_large[metric][model1], all_results_large[metric][model2])
                wilcoxon_results_large[metric].loc[model1, model2] = p_value
                wilcoxon_results_large[metric].loc[model2, model1] = p_value  # Fill the symmetric part of the matrix
                if p_value < 0.001:
                    pvalue_count_by_metric_large[metric]['< 0.001'] += 1
                elif 0.001 <= p_value <= 0.05:
                    pvalue_count_by_metric_large[metric]['0.001 - 0.05'] += 1
                else:
                    pvalue_count_by_metric_large[metric]['> 0.05'] += 1
pvalue_categories = {
    '< 0.001': 0,
    '0.001 - 0.05': 0,
    '> 0.05': 0
}
# Loop through each metric and perform pairwise Wilcoxon tests between model combinations
pvalue_count_by_metric_small = {metric: pvalue_categories.copy() for metric in metrics}
# Wilcoxon test for small datasets
for metric in metrics:
    for i, model1 in enumerate(combinations):
        for j, model2 in enumerate(combinations):
            if i < j:  # Perform the test for unique pairs
                stat, p_value = wilcoxon(all_results_small[metric][model1], all_results_small[metric][model2])
                wilcoxon_results_small[metric].loc[model1, model2] = p_value
                wilcoxon_results_small[metric].loc[model2, model1] = p_value  # Fill the symmetric part of the matrix
                # Categorize the p-value
                if p_value < 0.001:
                    pvalue_count_by_metric_small[metric]['< 0.001'] += 1
                elif 0.001 <= p_value <= 0.05:
                    pvalue_count_by_metric_small[metric]['0.001 - 0.05'] += 1
                else:
                    pvalue_count_by_metric_small[metric]['> 0.05'] += 1
'''
# Save the Wilcoxon test results for large and small datasets into an Excel file
output_path = 'C:/Users/BOUKA/OneDrive/Bureau/CODE/results_averaged/Wilcoxon_results_by_size.xlsx'
with pd.ExcelWriter(output_path) as writer:
    # Save large dataset results
    for metric, df in wilcoxon_results_large.items():
        df.to_excel(writer, sheet_name=f'Large_{metric}')
    # Save small dataset results
    for metric, df in wilcoxon_results_small.items():
        df.to_excel(writer, sheet_name=f'Small_{metric}')
'''
print("Wilcoxon test results for large and small datasets saved successfully!")

save_dir = "C:/Users/BOUKA/OneDrive/Bureau/CODE/results_averaged/"
# Convert p-value counts to DataFrame
pvalue_count_df_small = pd.DataFrame.from_dict(pvalue_count_by_metric_small, orient='index')
#pvalue_count_df_small.to_excel(f'{save_dir}small_pvalue_categorization_results.xlsx', index=True)


pvalue_count_df_large = pd.DataFrame.from_dict(pvalue_count_by_metric_large, orient='index')
#pvalue_count_df_large.to_excel(f'{save_dir}large_pvalue_categorization_results.xlsx', index=True)





# Friedman test
from scipy.stats import friedmanchisquare
# Perform Friedman test for large datasets group
friedman_results_large = []
metrics_=['Trtime']
for metric in metrics_:
    scores = all_results_large[metric].values  # Get the values (datasets x models)
    labels = all_results_large[metric].columns  # Model names

    # Perform Friedman test on the large dataset group
    stat, p_value = friedmanchisquare(*scores.T)
    friedman_results_large.append({'Metric': metric, 'Friedman stat': stat, 'p-value': p_value})
    print(f"Friedman test for large datasets ({metric}): stat={stat}, p-value={p_value}")
        # Plot the Critical Difference Diagram
    fig, ax = plot_critical_difference(
        scores, 
        labels, 
        test='wilcoxon',   # Use Wilcoxon test (or 'nemenyi')
        correction='holm', # Use Holm correction
        alpha=0.05,         # Significance level
        lower_better=True, # Higher scores are better
        reverse=True       # Lower rank on the right side
    )

    # Show the plot
    #plt.show()
    fig.set_size_inches(12, 8) 
    plt.subplots_adjust(left=0.1, right=0.2, top=0.9, bottom=0.8)  # You can adjust the values
    fig_name = f"All_CD_Plot_large_{metric}.png"  # Adjust this if the file names are different
    save_dir = os.path.join("C:/Users/BOUKA/OneDrive/Bureau/CODE/plots/", fig_name)
    # Optionally save the plot
    fig.savefig(save_dir, bbox_inches="tight")
    plt.show()

# Perform Friedman test for small datasets group
friedman_results_small = []
for metric in metrics_:
    scores = all_results_small[metric].values  # Get the values (datasets x models)
    labels = all_results_small[metric].columns  # Model names

    # Perform Friedman test on the small dataset group
    stat, p_value = friedmanchisquare(*scores.T)
    friedman_results_small.append({'Metric': metric, 'Friedman stat': stat, 'p-value': p_value})
    print(f"Friedman test for small datasets ({metric}): stat={stat}, p-value={p_value}")
        # Plot the Critical Difference Diagram
    fig, ax = plot_critical_difference(
        scores, 
        labels, 
        test='wilcoxon',   # Use Wilcoxon test (or 'nemenyi')
        correction='holm', # Use Holm correction
        alpha=0.05,         # Significance level
        lower_better=True, # Higher scores are better
        reverse=True       # Lower rank on the right side
    )

    # Show the plot
    fig.set_size_inches(12, 8) 
    plt.subplots_adjust(left=0.1, right=0.2, top=0.9, bottom=0.8)  # You can adjust the values
    fig_name = f"All_CD_Plot_small_{metric}.png"  # Adjust this if the file names are different
    save_dir = os.path.join("C:/Users/BOUKA/OneDrive/Bureau/CODE/plots/", fig_name)
    # Optionally save the plot
    fig.savefig(save_dir, bbox_inches="tight")
    plt.show()

    
'''
# Save the Friedman test results to Excel
output_path = 'C:/Users/BOUKA/OneDrive/Bureau/CODE/results_averaged/Friedman_test_by_size.xlsx'
with pd.ExcelWriter(output_path) as writer:
    # Convert to DataFrame and save large dataset results
    df_friedman_large = pd.DataFrame(friedman_results_large)
    df_friedman_large.to_excel(writer, sheet_name='Large_datasets')

    # Convert to DataFrame and save small dataset results
    df_friedman_small = pd.DataFrame(friedman_results_small)
    df_friedman_small.to_excel(writer, sheet_name='Small_datasets')

print("Friedman test results for large and small datasets saved successfully!")

##########################################################################################
##########################################################################################
##########################################################################################
import os
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

# Define the folder containing the result Excel files
folder_path = 'C:/Users/BOUKA/OneDrive/Bureau/CODE/results_test'

# Define the different combinations for IS, TSR, and IF
IS_variants = ['IS0', 'IS2']  # Interval Selection methods
metrics = ['Accuracy', 'F1', 'Auc', 'Balanced_accuracy', 'Trtime']
TSR_variants = ['TSR1', 'TSR2', 'TSR3']  # Time series representations
IF_variants = ['IF1', 'IF2', 'IF3', 'IF4', 'IF5']  # Interval features

# Define dataset groups
large_datasets_list = large_data_instances  # Replace with the actual list of large datasets
small_datasets_list = small_data_instances 


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
'''
# Display the averaged results
for IS_var in IS_variants:
    print(f"Results for {IS_var}:")
    for metric in metrics:
        print(f"\nAveraged {metric.capitalize()}:\n")
        print(results_dict[IS_var][metric])
'''
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

# Perform Friedman and Wilcoxon tests for a dataset group

def perform_tests(final_df_IS, final_df_TSR, final_df_IF, group_name):
    # Wilcoxon test for IS variable (IS0 vs. IS2)
    wilcoxon_results_IS = []
    for metric in metrics:
        stat, p_value = wilcoxon(
            final_df_IS[f'IS0_Averaged_{metric.capitalize()}'],
            final_df_IS[f'IS2_Averaged_{metric.capitalize()}']
        )
        wilcoxon_results_IS.append({'Metric': metric, 'Wilcoxon stat': stat, 'p-value': p_value})

    # Friedman test for TSR variable (TSR1, TSR2, TSR3)
    friedman_results_TSR = []
    for metric in metrics:
        stat, p_value = friedmanchisquare(
            final_df_TSR[f'TSR1_Averaged_{metric.capitalize()}'],
            final_df_TSR[f'TSR2_Averaged_{metric.capitalize()}'],
            final_df_TSR[f'TSR3_Averaged_{metric.capitalize()}']
        )
        friedman_results_TSR.append({'Metric': metric, 'Friedman stat': stat, 'p-value': p_value})

    # Friedman test for IF variable (IF1, IF2, IF3, IF4, IF5)
    friedman_results_IF = []
    for metric in metrics:
        stat, p_value = friedmanchisquare(
            final_df_IF[f'IF1_Averaged_{metric.capitalize()}'],
            final_df_IF[f'IF2_Averaged_{metric.capitalize()}'],
            final_df_IF[f'IF3_Averaged_{metric.capitalize()}'],
            final_df_IF[f'IF4_Averaged_{metric.capitalize()}'],
            final_df_IF[f'IF5_Averaged_{metric.capitalize()}']
        )
        friedman_results_IF.append({'Metric': metric, 'Friedman stat': stat, 'p-value': p_value})

    # Save results to Excel for this group
    output_path = f'C:/Users/BOUKA/OneDrive/Bureau/CODE/results_averaged/{group_name}_datasets_results.xlsx'
    with pd.ExcelWriter(output_path) as writer:
        pd.DataFrame(wilcoxon_results_IS).to_excel(writer, sheet_name='Wilcoxon_IS')
        pd.DataFrame(friedman_results_TSR).to_excel(writer, sheet_name='Friedman_TSR')
        pd.DataFrame(friedman_results_IF).to_excel(writer, sheet_name='Friedman_IF')

    print(f"Tests results saved for {group_name} datasets!")


final_df_IF_large = final_df_IF[final_df_IF['Dataset'].isin(large_datasets_list)]
print(f"shape of large final_df_IF_large {final_df_IF_large.shape}")
final_df_IF_small = final_df_IF[final_df_IF['Dataset'].isin(small_datasets_list)]
print(f"shape of final_df_IF_small {final_df_IF_small.shape}")

final_df_TSR_large = final_df_TSR[final_df_TSR['Dataset'].isin(large_datasets_list)]
final_df_TSR_small = final_df_TSR[final_df_TSR['Dataset'].isin(small_datasets_list)]

final_df_IS_large = final_df_IS[final_df_IS['Dataset'].isin(large_datasets_list)]
final_df_IS_small = final_df_IS[final_df_IS['Dataset'].isin(small_datasets_list)]
perform_tests(final_df_IS_large, final_df_TSR_large, final_df_IF_large, 'large')
perform_tests(final_df_IS_small, final_df_TSR_small, final_df_IF_small, 'small')
'''

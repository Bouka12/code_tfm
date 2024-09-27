import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


def categorize_pvalue(pvalue):
    if pvalue < 0.001:
        return '< 0.001'
    elif 0.001 <= pvalue < 0.05:
        return '0.001 - 0.05'
    else :
        return '> 0.05 '


# Assuming row_labels and col_labels are given as follows:
row_labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]
col_labels = ['IS0_TSR1_IF1', 'IS0_TSR1_IF2', 'IS0_TSR1_IF3', 'IS0_TSR1_IF4', 'IS0_TSR1_IF5', 
              'IS0_TSR2_IF1', 'IS0_TSR2_IF2', 'IS0_TSR2_IF3', 'IS0_TSR2_IF4', 'IS0_TSR2_IF5',
              'IS0_TSR3_IF1', 'IS0_TSR3_IF2', 'IS0_TSR3_IF3', 'IS0_TSR3_IF4', 'IS0_TSR3_IF5',
              'IS2_TSR1_IF1', 'IS2_TSR1_IF2', 'IS2_TSR1_IF3', 'IS2_TSR1_IF4', 'IS2_TSR1_IF5',
              'IS2_TSR2_IF1', 'IS2_TSR2_IF2', 'IS2_TSR2_IF3', 'IS2_TSR2_IF4', 'IS2_TSR2_IF5',
              'IS2_TSR3_IF1', 'IS2_TSR3_IF2', 'IS2_TSR3_IF3', 'IS2_TSR3_IF4', 'IS2_TSR3_IF5']

# Create a mapping from row index to model names using the col_labels
index_to_model = {i: col_labels[i] for i in row_labels}
# Assuming 'data' is the matrix with Wilcoxon p-values
sheet_names = ['Accuracy', 'F1', 'Auc', 'Balanced_accuracy', 'Trtime']
for sheet_name in sheet_names:

    data = pd.read_excel('C:/Users/BOUKA/OneDrive/Bureau/CODE/results_averaged/All_models_Wilcoxon_results.xlsx',sheet_name=sheet_name)  # Load the CSV file


    # Convert all the data to numeric, keeping NaN for any non-numeric values
    df_numeric = data.apply(pd.to_numeric, errors='coerce')
    #print(f"df_numeric:{df_numeric}")
    # Define p-value categories (you can customize this)

    # Initialize an empty list to store the results
    results = []

    # Loop through the DataFrame to create pairs of models and their corresponding p-value category
    #row_labels = []
    #col_labels = []
    for row_label, row in df_numeric.iterrows():
        row_labels.append(row_label)
        for col_label, pvalue in row.items():
            col_labels.append(col_label)
            if pd.notna(pvalue):  # Skip NaN values
                model_1 = index_to_model[row_label]  # Map the row label to the actual model name
                category = categorize_pvalue(pvalue)  # Apply your categorization function
                results.append([model_1, col_label, pvalue, category])

    # Create a DataFrame with the categorized results
    categorized_df = pd.DataFrame(results, columns=['Model 1', 'Model 2', 'p-value', 'Category'])

    # Optionally, sort the table by p-value or category
    categorized_df.sort_values(by='p-value', inplace=True)

    # Show or save the result
    print(categorized_df)
    #print(f"row_labels : {row_labels}")
    col_labels_corrected = col_labels[1:31]
    #print(f"col_labels : {col_labels_corrected}")
    # C:/Users\/BOUKA/OneDrive/Bureau/CODE/results_averaged
    categorized_df.to_excel(f'C:/Users/BOUKA/OneDrive/Bureau/CODE/results_averaged/All_{sheet_name}_categorized_pvalues.xlsx', index=False)
    # Count the number of pairs in each category
    category_counts = categorized_df['Category'].value_counts().reset_index()

    # Rename columns for better understanding
    category_counts.columns = ['Category', 'Number of Pairs']

    # Sort by Category (optional, depending on how you want to display)
    category_counts.sort_values(by='Category', inplace=True)

    # Show the result
    print(category_counts)

    # Optionally, save the table to an Excel or CSV file
    category_counts.to_excel(f'C:/Users/BOUKA/OneDrive/Bureau/CODE/results_averaged/All_{sheet_name}_pvalue_category_counts.xlsx', index=False)


import pandas as pd

# Replace 'data.csv' with the path to your CSV file
df = pd.read_csv('data.csv')

# List of all metrics
metrics = ['F1', 'MCC', 'ROC-AUC', 'PRC-AUC']

# List of all tasks (antibiotics)
tasks = df['Task'].unique()

# Initialize a DataFrame to store ranks
rank_df = pd.DataFrame()

# Calculate ranks for each task and metric
for task in tasks:
    task_df = df[df['Task'] == task]
    for metric in metrics:
        # Rank models for the current task and metric
        task_df = task_df.sort_values(by=metric, ascending=False)
        task_df[f'{metric}_Rank'] = range(1, len(task_df) + 1)
    rank_df = pd.concat([rank_df, task_df], ignore_index=True)

# Melt the DataFrame to long format for easy grouping
rank_melted = rank_df.melt(
    id_vars=['Model', 'Task'],
    value_vars=[f'{metric}_Rank' for metric in metrics],
    var_name='Metric',
    value_name='Rank'
)

# Clean the Metric column
rank_melted['Metric'] = rank_melted['Metric'].str.replace('_Rank', '')

# Calculate average rank for each model and metric
average_ranks = rank_melted.groupby(['Model', 'Metric'])['Rank'].mean().reset_index()

# Pivot the DataFrame for a cleaner view
average_ranks_pivot = average_ranks.pivot(index='Model', columns='Metric', values='Rank')

# Sort the models based on average rank (optional)
average_ranks_pivot = average_ranks_pivot.sort_values(by=metrics)

# Display the average ranks
print(average_ranks_pivot)

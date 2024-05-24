import pandas as pd
import numpy as np

def custom_train_test_split(data, test_size=0.2, random_state=None):
    """
    Custom function to split a pandas DataFrame into train and test DataFrames,
    ensuring that subject_id and hadm_id are not split between train and test sets.
    
    Parameters:
    - data: pandas DataFrame
        The DataFrame to be split.
    - test_size: float, default=0.2
        The proportion of the dataset to include in the test split.
    - random_state: int or None, default=None
        Controls the randomness of the data splitting. Pass an integer for reproducible results.
    
    Returns:
    - train_data: pandas DataFrame
        The training DataFrame.
    - test_data: pandas DataFrame
        The testing DataFrame.
    """
    # Group by subject_id and hadm_id
    unique_groups = data[['subject_id', 'hadm_id']].drop_duplicates()

    # Shuffle the groups
    if random_state is not None:
        np.random.seed(random_state)
    shuffled_groups = unique_groups.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Calculate the index to split the groups
    split_index = int(len(shuffled_groups) * (1 - test_size))

    # Split the groups into train and test
    train_groups = shuffled_groups.iloc[:split_index]
    test_groups = shuffled_groups.iloc[split_index:]

    # Use boolean indexing to filter the original data based on the split groups
    train_data = data[data[['subject_id', 'hadm_id']].apply(tuple, axis=1).isin(train_groups.apply(tuple, axis=1))]
    test_data = data[data[['subject_id', 'hadm_id']].apply(tuple, axis=1).isin(test_groups.apply(tuple, axis=1))]

    return train_data, test_data
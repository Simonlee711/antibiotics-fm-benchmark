import pickle
def save_pickle(results, filename):
    """
    Creates a new pickle file

    results: the name of the variable
    filename: the name of the file
    """
    filename = f"{filename}.pickle"

    file_path=f'./results/{filename}'
    with open(file_path, 'wb') as file:
        pickle.dump(results, file)

    print(f"The variable {filename} has been saved successfully.")

def load_pickle(filename):
    """
    Loads in the pickle file

    filename: the name of the file
    """
    file_path=f'./results/{filename}.pickle'
    loaded_data = None
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)

    print(f"The variable {filename} has been loaded successfully.")

    return loaded_data

def compute_average_rank(antibiotics, dictionaries, metric):
    """
    Computes the average rank for each model across all antibiotics based on AUROC values.

    Parameters:
    - antibiotics: List of antibiotics
    - dictionaries: List of tuples containing (dictionary, model_name)

    Returns:
    - Dictionary with each model and their average rank across all antibiotics
    """
    model_ranks = {name: [] for _, name in dictionaries}  # Initialize dictionary to store ranks for each model

    # Loop through each antibiotic
    for antibiotic in antibiotics:
        # Sort the dictionaries based on AUROC values for each antibiotic, descending order (best to worst)
        sorted_dictionaries = sorted(dictionaries, key=lambda x: x[0][antibiotic]['Test Metrics'][metric], reverse=True)
        
        # Record the rank of each model for this antibiotic
        for rank, (_, name) in enumerate(sorted_dictionaries, start=1):
            model_ranks[name].append(rank)

    # Compute the average rank for each model
    average_ranks = {name: sum(ranks) / len(ranks) for name, ranks in model_ranks.items()}
    sorted_average_ranks = dict(sorted(average_ranks.items(), key=lambda item: item[1]))

    return sorted_average_ranks
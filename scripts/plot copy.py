import matplotlib.pyplot as plt
import numpy as np
def plot_roc_curves(antibiotics, dictionaries, model_colors, figsize=(20, 20)):
    """
    Plots ROC curves for multiple models and antibiotics.

    Parameters:
    - antibiotics: List of antibiotics
    - dictionaries: List of tuples containing (dictionary, model_name)
    - model_colors: Dictionary of colors keyed by model_name
    - figsize: Tuple for figure size

    Returns:
    - Matplotlib figure with ROC curves
    """
    # Calculate number of rows and columns needed for subplots
    n = len(antibiotics)
    cols = 5
    rows = n // cols + (1 if n % cols > 0 else 0)
    
    # Create figure and axes objects
    fig, axs = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    
    # Loop through all antibiotics and plot each on a subplot
    for i, antibiotic in enumerate(antibiotics):
        row, col = divmod(i, cols)
        ax = axs[row, col]
        
        for dictionary, name in dictionaries:
            results = dictionary[antibiotic]['Test Metrics']
            fpr, tpr = results['fpr'], results['tpr']
            roc_auc = results['ROC AUC']
            
            # Assuming calculate_confidence_interval is a function that returns lower and upper confidence intervals
            ci_lower_tpr, ci_upper_tpr = calculate_confidence_interval(tpr)
            
            # Plot ROC curve
            line_color = model_colors[name]
            ax.plot(fpr, tpr, label=f'{name} (ROC AUC = {roc_auc:.4f})', color=line_color)
            
            # Fill between for confidence intervals using the same color
            ax.fill_between(fpr, ci_lower_tpr, ci_upper_tpr, color=line_color, alpha=0.3)
        
        # Set plot details
        ax.legend(loc='upper left')
        ax.set_title(f'{antibiotic} - ROC Curves')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    
def plot_auprc_curves(antibiotics, dictionaries, model_colors, figsize=(20, 20)):
    """
    Plots AUPRC curves for multiple models and antibiotics.

    Parameters:
    - antibiotics: List of antibiotics
    - dictionaries: List of tuples containing (dictionary, model_name)
    - model_colors: Dictionary of colors keyed by model_name
    - figsize: Tuple for figure size

    Returns:
    - Matplotlib figure with AUPRC curves
    """
    # Calculate number of rows and columns needed for subplots
    n = len(antibiotics)
    cols = 5
    rows = n // cols + (1 if n % cols > 0 else 0)
    
    # Create figure and axes objects
    fig, axs = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    
    # Loop through all antibiotics and plot each on a subplot
    for i, antibiotic in enumerate(antibiotics):
        row, col = divmod(i, cols)
        ax = axs[row, col]
        
        for dictionary, name in dictionaries:
            results = dictionary[antibiotic]['Test Metrics']
            precision, recall = results['precision'], results['recall']
            auprc = results['auprc']
            
            # Assuming calculate_confidence_interval is a function that returns lower and upper confidence intervals
            ci_lower, ci_upper = calculate_confidence_interval(precision)
            
            # Plot Precision-Recall curve
            line_color = model_colors[name]
            ax.plot(recall, precision, linestyle='--', color=line_color, label=f'{name} (PR AUC = {auprc:.4f})')
            
            # Fill between for confidence intervals using the same color
            ax.fill_between(recall, ci_lower, ci_upper, color=line_color, alpha=0.3)
        
        # Set plot details
        ax.legend(loc='best')
        ax.set_title(f'{antibiotic} - AUPRC Curves')
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def calculate_confidence_interval(tpr):
    """
    Dummy function to calculate confidence intervals for TPR.
    Replace with actual calculation method.
    """
    ci_lower = tpr - 0.05  # Dummy values for example purposes
    ci_upper = tpr + 0.05
    return np.maximum(0, ci_lower), np.minimum(1, ci_upper)


def print_name():
    return "Hello My name is Simon"


l = print_name()
print(l)


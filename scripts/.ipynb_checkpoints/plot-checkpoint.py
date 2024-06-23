import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

def plot_roc_curves(antibiotics, dictionaries, model_colors, figsize=(20, 20)):
    """
    Plots ROC curves for multiple models and antibiotics with confidence intervals.

    Parameters:
    - antibiotics: List of antibiotics
    - dictionaries: List of tuples containing (dictionary, model_name)
    - model_colors: Dictionary of colors keyed by model_name
    - figsize: Tuple for figure size

    Returns:
    - Matplotlib figure with ROC curves
    """
    n = len(antibiotics)
    cols = 4
    rows = n // cols + (1 if n % cols > 0 else 0)
    
    fig, axs = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    
    for i, antibiotic in enumerate(antibiotics):
        row, col = divmod(i, cols)
        ax = axs[row, col]
        
        all_tpr = [dictionary[antibiotic]['Test Metrics']['tpr'] for dictionary, _ in dictionaries]
        all_fpr = [dictionary[antibiotic]['Test Metrics']['fpr'] for dictionary, _ in dictionaries]
        
        mean_fpr = np.linspace(0, 1, 100)
        interp_tpr = [np.interp(mean_fpr, fpr, tpr) for fpr, tpr in zip(all_fpr, all_tpr)]
        
        ci_lower, ci_upper = calculate_confidence_interval_curves(interp_tpr)
        
        for idx, (dictionary, name) in enumerate(dictionaries):
            tpr, fpr = dictionary[antibiotic]['Test Metrics']['tpr'], dictionary[antibiotic]['Test Metrics']['fpr']
            roc_auc = dictionary[antibiotic]['Test Metrics']['ROC AUC']
            
            line_color = model_colors[name]
            ax.plot(fpr, tpr, linestyle='--', color=line_color, label=f'{name} (ROC AUC = {roc_auc:.4f})')
            ax.fill_between(mean_fpr, ci_lower, ci_upper, color=line_color, alpha=0.3)
        
        ax.legend(loc='best')
        ax.set_title(f'{antibiotic} - ROC Curves')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])

    plt.tight_layout()
    plt.show()


def plot_auprc_curves(antibiotics, dictionaries, model_colors, figsize=(20, 20)):
    """
    Plots AUPRC curves for multiple models and antibiotics with confidence intervals.

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
    cols = 4
    rows = n // cols + (1 if n % cols > 0 else 0)
    
    # Create figure and axes objects
    fig, axs = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    
    # Loop through all antibiotics and plot each on a subplot
    for i, antibiotic in enumerate(antibiotics):
        row, col = divmod(i, cols)
        ax = axs[row, col]
        
        # Gather all precision data for this antibiotic to calculate confidence intervals
        all_precision = [dictionary[antibiotic]['Test Metrics']['precision'] for dictionary, _ in dictionaries]
        all_recall = [dictionary[antibiotic]['Test Metrics']['recall'] for dictionary, _ in dictionaries]

        mean_recall = np.linspace(0, 1, 100)  # Common recall grid
        interp_precision = [np.interp(mean_recall, recall, precision) for recall, precision in zip(all_recall, all_precision)]
        
        ci_lower, ci_upper = calculate_confidence_interval_curves(interp_precision)
        
        for idx, (dictionary, name) in enumerate(dictionaries):
            results = dictionary[antibiotic]['Test Metrics']
            precision, recall = results['precision'], results['recall']
            auprc = results['PRC AUC']
            
            # Plot Precision-Recall curve
            line_color = model_colors[name]
            ax.plot(recall, precision, linestyle='--', color=line_color, label=f'{name} (PR AUC = {auprc:.4f})')
        
        # Fill between for confidence intervals using a neutral color
        ax.fill_between(mean_recall, ci_lower, ci_upper, color=line_color, alpha=0.3)
        
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


def calculate_confidence_interval_curves(values):
    """
    Calculates the 95% confidence intervals for an array of value arrays (e.g., multiple TPR curves).

    Parameters:
    - values: List of numpy arrays (each array is a curve like TPR or precision)

    Returns:
    - Tuple of numpy arrays (ci_lower, ci_upper) for each point on the curve
    """
    values = np.array(values)
    mean_values = np.mean(values, axis=0)
    sem = np.std(values, axis=0, ddof=1) / np.sqrt(values.shape[0])  # Standard Error of the Mean
    confidence_level = 0.95
    degrees_of_freedom = values.shape[1] - 1

    # Calculate t-critical for two-tailed test
    t_critical = t.ppf((1 + confidence_level) / 2, degrees_of_freedom)

    # Calculate margin of error for each point
    margin_of_error = t_critical * sem

    # Lower and upper confidence intervals
    ci_lower = mean_values - margin_of_error
    ci_upper = mean_values + margin_of_error

    return ci_lower, ci_upper


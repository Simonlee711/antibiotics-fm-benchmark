import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

def plot_roc_curves(antibiotics, dictionaries, model_colors, figsize=(20, 20)):
    """
    Plots ROC curves for multiple models and antibiotics with individual confidence intervals.

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
        
        for dictionary, name in dictionaries:
            tpr = dictionary[antibiotic]['Test Metrics']['tpr']
            fpr = dictionary[antibiotic]['Test Metrics']['fpr']
            roc_auc = dictionary[antibiotic]['Test Metrics']['ROC AUC']
            ci = dictionary[antibiotic]['Confidence Intervals']['ROC AUC']['95% CI']
            
            line_color = model_colors[name]
            ax.plot(fpr, tpr, linestyle='--', color=line_color, label=f'{name} (ROC AUC = {roc_auc:.4f})')
            
            ci1, ci2 = calculate_confidence_interval_curves(tpr, ci)
            ax.fill_between(fpr, ci1, ci2, color=line_color, alpha=0.3)
        
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
    Plots AUPRC curves for multiple models and antibiotics, each with their individual confidence intervals.

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
        
        for dictionary, name in dictionaries:
            results = dictionary[antibiotic]['Test Metrics']
            precision, recall = results['precision'], results['recall']
            auprc = results['PRC AUC']
            ci = dictionary[antibiotic]['Confidence Intervals']['PRC AUC']['95% CI']
            
            # Plot Precision-Recall curve
            line_color = model_colors[name]
            ax.plot(recall, precision, linestyle='--', color=line_color, label=f'{name} (PR AUC = {auprc:.4f})')
            
            ci1, ci2 = calculate_confidence_interval_curves(precision, ci)
            
            # Fill between for confidence intervals using a neutral color
            ax.fill_between(recall, ci1, ci2, color=line_color, alpha=0.3)
        
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



def calculate_confidence_interval_curves(mean_values, errors):
    """
    Calculates the 95% confidence intervals for an array of value arrays (e.g., multiple TPR curves).

    Parameters:
    - values: List of numpy arrays (each array is a curve like TPR or precision)

    Returns:
    - Tuple of numpy arrays (ci_lower, ci_upper) for each point on the curve
    """
    # Lower and upper confidence intervals
    ci_lower = mean_values - ((errors[1] - errors[0])/2)
    ci_upper = mean_values + ((errors[1] - errors[0])/2)

    return ci_lower, ci_upper












def plot_bar_auprc(antibiotics, dictionaries, model_colors, figsize=(20, 20)):
    """
    Plots bar charts for AUPRC values for multiple models and antibiotics, including error bars for confidence intervals,
    sorted from best to worst.

    Parameters:
    - antibiotics: List of antibiotics
    - dictionaries: List of tuples containing (dictionary, model_name)
    - model_colors: Dictionary of colors keyed by model_name
    - figsize: Tuple for figure size

    Returns:
    - Matplotlib figure with AUPRC bar charts
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
        
        # Sort the dictionaries based on AUPRC values for each antibiotic
        sorted_dictionaries = sorted(dictionaries, key=lambda x: x[0][antibiotic]['Test Metrics']['auprc'])
        
        # Initialize position for bars
        y_positions = range(len(sorted_dictionaries))

        for pos, (dictionary, name) in zip(y_positions, sorted_dictionaries):
            auprc = dictionary[antibiotic]['Test Metrics']['auprc']
            ci = dictionary[antibiotic]['Confidence Intervals']['PRC AUC']['95% CI']
            
            # Calculate error values for error bars
            error = [[auprc - ci[0]], [ci[1] - auprc]]  # Error bars format for matplotlib (asymmetric)
            
            # Plot bar with error bars
            line_color = model_colors[name]
            ax.barh(pos, auprc, color=line_color, xerr=error, capsize=5,
                    label=f'{name} (PR AUC = {auprc:.4f})')
        
        # Set plot details
        ax.set_yticks(y_positions)
        ax.set_yticklabels([name for _, name in sorted_dictionaries])
        ax.legend(loc=4)
        ax.set_title(f'{antibiotic} - AUPRC')
        ax.set_xlabel('AUPRC')
        ax.set_xlim([0, 1])

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

    
def plot_bar_auroc(antibiotics, dictionaries, model_colors, figsize=(20, 20)):
    """
    Plots bar charts for AUPRC values for multiple models and antibiotics, including error bars for confidence intervals,
    sorted from best to worst.

    Parameters:
    - antibiotics: List of antibiotics
    - dictionaries: List of tuples containing (dictionary, model_name)
    - model_colors: Dictionary of colors keyed by model_name
    - figsize: Tuple for figure size

    Returns:
    - Matplotlib figure with AUPRC bar charts
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
        
        # Sort the dictionaries based on AUPRC values for each antibiotic
        sorted_dictionaries = sorted(dictionaries, key=lambda x: x[0][antibiotic]['Test Metrics']['ROC AUC'])
        
        # Initialize position for bars
        y_positions = range(len(sorted_dictionaries))

        for pos, (dictionary, name) in zip(y_positions, sorted_dictionaries):
            auprc = dictionary[antibiotic]['Test Metrics']['ROC AUC']
            ci = dictionary[antibiotic]['Confidence Intervals']['ROC AUC']['95% CI']
            
            # Calculate error values for error bars
            error = [[auprc - ci[0]], [ci[1] - auprc]]  # Error bars format for matplotlib (asymmetric)
            
            # Plot bar with error bars
            line_color = model_colors[name]
            ax.barh(pos, auprc, color=line_color, xerr=error, capsize=5,
                    label=f'{name} (ROC AUC = {auprc:.4f})')
        
        # Set plot details
        ax.set_yticks(y_positions)
        ax.set_yticklabels([name for _, name in sorted_dictionaries])
        ax.legend(loc=4)
        ax.set_title(f'{antibiotic} - AUROC')
        ax.set_xlabel('AUROC')
        ax.set_xlim([0, 1])

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    
    # import matplotlib.pyplot as plt

# def plot_bar_auroc(antibiotics, dictionaries, model_colors, figsize=(24, 20)):  # Increased width for better layout
#     """
#     Plots bar charts for AUROC values for multiple models and antibiotics, including error bars for confidence intervals,
#     sorted from best to worst.

#     Parameters:
#     - antibiotics: List of antibiotics
#     - dictionaries: List of tuples containing (dictionary, model_name)
#     - model_colors: Dictionary of colors keyed by model_name
#     - figsize: Tuple for figure size

#     Returns:
#     - Matplotlib figure with AUROC bar charts
#     """
#     # Calculate number of rows and columns needed for subplots
#     n = len(antibiotics)
#     cols = 4
#     rows = n // cols + (1 if n % cols > 0 else 0)

#     # Create figure and axes objects
#     fig, axs = plt.subplots(rows, cols, figsize=figsize, squeeze=False)

#     # Loop through all antibiotics and plot each on a subplot
#     for i, antibiotic in enumerate(antibiotics):
#         row, col = divmod(i, cols)
#         ax = axs[row, col]
        
#         # Sort the dictionaries based on AUROC values for each antibiotic
#         sorted_dictionaries = sorted(dictionaries, key=lambda x: x[0][antibiotic]['Test Metrics']['ROC AUC'], reverse=True)
        
#         # Initialize position for bars
#         y_positions = range(len(sorted_dictionaries))

#         for pos, (dictionary, name) in zip(y_positions, sorted_dictionaries):
#             auroc = dictionary[antibiotic]['Test Metrics']['ROC AUC']
#             ci = dictionary[antibiotic]['Confidence Intervals']['ROC AUC']['95% CI']
            
#             # Calculate error values for error bars
#             error = [[auroc - ci[0]], [ci[1] - auroc]]  # Error bars format for matplotlib (asymmetric)
            
#             # Plot bar with error bars
#             line_color = model_colors[name]
#             ax.barh(pos, auroc, color=line_color, xerr=error, capsize=5,
#                     label=f'{name} (ROC AUC = {auroc:.4f})')
        
#         # Set plot details
#         ax.set_yticks(y_positions)
#         ax.set_yticklabels([name for _, name in sorted_dictionaries])
#         # Place the legend to the right of the plot
#         ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
#         ax.set_title(f'{antibiotic} - AUROC')
#         ax.set_xlabel('AUROC')
#         ax.set_xlim([0, 1])

#     # Adjust layout to prevent overlap
#     plt.tight_layout()
#     plt.show()


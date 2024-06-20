import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc
from lightgbm import LGBMClassifier

def evaluate_antibiotics(X_train, X_test, train, test, antibiotics):
    """
    Function to train and evaluate a model for each antibiotic in the list.

    Parameters:
    - X_train: Features for the training set
    - X_test: Features for the testing set
    - train: Training dataset containing the targets
    - test: Testing dataset containing the targets
    - antibiotics: List of antibiotics to evaluate

    Returns:
    - A dictionary containing evaluation results for each antibiotic.
    """
    results = {}
    for antibiotic in antibiotics:
        print(f"Evaluating: {antibiotic}")
        y_train = train[antibiotic].astype(int)
        y_test = test[antibiotic].astype(int)
        
        # Initialize and fit the model
        model = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=30)
        model.fit(X_train, y_train)
        
        # Predict on test set and calculate probabilities
        y_test_proba = model.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
        f1_scores = 2 * recall * precision / (recall + precision)
        f1_scores = np.nan_to_num(f1_scores)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
        
        # Evaluate the model
        mcc_test = matthews_corrcoef(y_test, y_test_pred)
        roc_auc_test = roc_auc_score(y_test, y_test_proba)
        prc_auc_test = average_precision_score(y_test, y_test_proba)
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        auprc = auc(recall, precision)
        
        # Store results
        results[antibiotic] = {
            'Optimal Threshold': optimal_threshold,
            'Test Metrics': {
                'F1 Score': optimal_f1,
                'Matthews Correlation Coefficient': mcc_test,
                'ROC AUC': roc_auc_test,
                'PRC AUC': prc_auc_test,
                'fpr': fpr,
                'tpr': tpr,
                'auprc': auprc,
                'precision': precision,
                'recall': recall
            }
        }
    return results

import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score, average_precision_score, precision_recall_curve, roc_curve, auc
from sklearn.utils import resample
from lightgbm import LGBMClassifier
from tqdm import tqdm

def evaluate_antibiotics_with_confidence_intervals(X_train, X_test, train, test, antibiotics, n_bootstraps=1000):
    """
    Function to train and evaluate a model for each antibiotic in the list, including confidence intervals
    for metrics using bootstrapping.

    Parameters:
    - X_train: Features for the training set
    - X_test: Features for the testing set
    - train: Training dataset containing the targets
    - test: Testing dataset containing the targets
    - antibiotics: List of antibiotics to evaluate
    - n_bootstraps: Number of bootstrap samples to use for confidence intervals

    Returns:
    - A dictionary containing evaluation results and confidence intervals for each antibiotic.
    """
    results = {}
    for antibiotic in tqdm(antibiotics,desc="Iterating through Antibiotics Progress: "):
        y_train = train[antibiotic].astype(int).reset_index(drop=True)
        y_test = test[antibiotic].astype(int).reset_index(drop=True)
        
        # Initialize and fit the model
        model = LGBMClassifier(n_estimators=1000, learning_rate=0.05, num_leaves=30)
        model.fit(X_train, y_train)
        
        # Predict on test set and calculate probabilities
        y_test_proba = model.predict_proba(X_test)[:, 1]

        # Initial evaluation
        precision, recall, thresholds = precision_recall_curve(y_test, y_test_proba)
        f1_scores = 2 * recall * precision / (recall + precision)
        f1_scores = np.nan_to_num(f1_scores)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
        mcc_test = matthews_corrcoef(y_test, y_test_pred)
        roc_auc_test = roc_auc_score(y_test, y_test_proba)
        prc_auc_test = average_precision_score(y_test, y_test_proba)
        fpr, tpr, _ = roc_curve(y_test, y_test_proba)
        auprc = auc(recall, precision)

        # Bootstrap confidence intervals
        roc_aucs = []
        prc_aucs = []
        f1_scores_list = []

        for _ in range(n_bootstraps):
            indices = resample(np.arange(len(y_test)), replace=True)
            y_test_resampled = y_test[indices]
            y_test_proba_resampled = y_test_proba[indices]

            roc_aucs.append(roc_auc_score(y_test_resampled, y_test_proba_resampled))
            pr, rc, _ = precision_recall_curve(y_test_resampled, y_test_proba_resampled)
            prc_aucs.append(auc(rc, pr))
            f1 = 2 * rc * pr / (np.maximum(rc + pr, np.finfo(float).eps))
            f1_scores_list.append(np.max(f1))

        # Store results including confidence intervals
        results[antibiotic] = {
            'Optimal Threshold': optimal_threshold,
            'Test Metrics': {
                'F1 Score': optimal_f1,
                'Matthews Correlation Coefficient': mcc_test,
                'ROC AUC': roc_auc_test,
                'PRC AUC': prc_auc_test,
                'fpr': fpr,
                'tpr': tpr,
                'auprc': auprc,
                'precision': precision,
                'recall': recall
            },
            'Confidence Intervals': {
                'ROC AUC': {'Mean': np.mean(roc_aucs), '95% CI': np.percentile(roc_aucs, [2.5, 97.5])},
                'PRC AUC': {'Mean': np.mean(prc_aucs), '95% CI': np.percentile(prc_aucs, [2.5, 97.5])},
                'F1 Score': {'Mean': np.mean(f1_scores_list), '95% CI': np.percentile(f1_scores_list, [2.5, 97.5])}
            }
        }
    
    return results

def print_results(results):
    # Print results
    for antibiotic, res in results.items():
        print(f"Results for {antibiotic}:")

        # Calculate mean and confidence interval half-width for F1 score
        f1_mean = res['Confidence Intervals']['F1 Score']['Mean']
        f1_ci_lower = res['Confidence Intervals']['F1 Score']['95% CI'][0]
        f1_ci_upper = res['Confidence Intervals']['F1 Score']['95% CI'][1]
        f1_error = (f1_ci_upper - f1_ci_lower) / 2

        # Calculate mean and confidence interval half-width for ROC AUC
        roc_auc_mean = res['Confidence Intervals']['ROC AUC']['Mean']
        roc_auc_ci_lower = res['Confidence Intervals']['ROC AUC']['95% CI'][0]
        roc_auc_ci_upper = res['Confidence Intervals']['ROC AUC']['95% CI'][1]
        roc_auc_error = (roc_auc_ci_upper - roc_auc_ci_lower) / 2

        # Calculate mean and confidence interval half-width for PRC AUC
        prc_auc_mean = res['Confidence Intervals']['PRC AUC']['Mean']
        prc_auc_ci_lower = res['Confidence Intervals']['PRC AUC']['95% CI'][0]
        prc_auc_ci_upper = res['Confidence Intervals']['PRC AUC']['95% CI'][1]
        prc_auc_error = (prc_auc_ci_upper - prc_auc_ci_lower) / 2

        # Print the metrics with confidence intervals
        print(f"  Test - F1: {f1_mean:.4f} +/- {f1_error:.4f}, MCC: {res['Test Metrics']['Matthews Correlation Coefficient']:.4f}, "
              f"ROC-AUC: {roc_auc_mean:.4f} +/- {roc_auc_error:.4f}, PRC-AUC: {prc_auc_mean:.4f} +/- {prc_auc_error:.4f}")




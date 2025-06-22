"""
Model analysis script for SST-2 sentiment analysis project.

Topics covered:
- Algorithm Analysis (capacity, under/overfitting, learning curves, regularization, CV)
"""
import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, learning_curve, validation_curve
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import argparse
from utils import PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR, load_pickle, save_pickle

def plot_learning_curve(estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10),
                        title="Learning Curve", save_path=None):
    """
    Plot a learning curve for a model.
    
    Topics: Algorithm Analysis (learning curves, under/overfitting)
    """
    plt.figure(figsize=(10, 6))
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring='accuracy', shuffle=True, random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="orange")
    plt.plot(train_sizes, train_mean, 'o-', color="blue", label="Training Score")
    plt.plot(train_sizes, test_mean, 'o-', color="orange", label="Cross-Validation Score")
    
    plt.title(title)
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Learning curve saved to {save_path}")
    
    plt.close()
    
    return train_sizes, train_mean, test_mean

def plot_validation_curve(estimator, X, y, param_name, param_range, cv=5, n_jobs=-1,
                          title="Validation Curve", save_path=None):
    """
    Plot a validation curve for a parameter.
    
    Topics: Algorithm Analysis (regularization, capacity)
    """
    plt.figure(figsize=(10, 6))
    
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, n_jobs=n_jobs, scoring='accuracy'
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color="blue")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color="orange")
    plt.plot(param_range, train_mean, 'o-', color="blue", label="Training Score")
    plt.plot(param_range, test_mean, 'o-', color="orange", label="Cross-Validation Score")
    
    plt.title(title)
    plt.xlabel(param_name)
    plt.ylabel("Accuracy")
    plt.legend(loc="best")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Validation curve saved to {save_path}")
    
    plt.close()
    
    return param_range, train_mean, test_mean

def compare_regularization(X_train, y_train, X_val, y_val, cv=5):
    """
    Compare L1 vs L2 regularization in Logistic Regression.
    
    Topics: Algorithm Analysis (regularization)
    """
    print("\nComparing L1 vs L2 regularization in Logistic Regression...")
    
    # Create models with different regularization
    C_values = [0.001, 0.01, 0.1, 1, 10, 100]
    
    l1_scores = []
    l2_scores = []
    
    for C in C_values:
        # L1 regularization
        l1_model = LogisticRegression(C=C, penalty='l1', solver='liblinear', max_iter=1000, random_state=42)
        l1_model.fit(X_train, y_train)
        l1_score = accuracy_score(y_val, l1_model.predict(X_val))
        l1_scores.append(l1_score)
        
        # L2 regularization
        l2_model = LogisticRegression(C=C, penalty='l2', max_iter=1000, random_state=42)
        l2_model.fit(X_train, y_train)
        l2_score = accuracy_score(y_val, l2_model.predict(X_val))
        l2_scores.append(l2_score)
        
        print(f"C={C}: L1 accuracy = {l1_score:.4f}, L2 accuracy = {l2_score:.4f}")
    
    # Plot comparison
    plt.figure(figsize=(10, 6))
    plt.plot(C_values, l1_scores, 'bo-', label='L1 Regularization')
    plt.plot(C_values, l2_scores, 'ro-', label='L2 Regularization')
    plt.xscale('log')
    plt.xlabel('C (inverse of regularization strength)')
    plt.ylabel('Validation Accuracy')
    plt.title('L1 vs L2 Regularization in Logistic Regression')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    os.makedirs(RESULTS_DIR, exist_ok=True)
    save_path = os.path.join(RESULTS_DIR, 'l1_vs_l2_regularization.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f"Regularization comparison plot saved to {save_path}")
    
    return C_values, l1_scores, l2_scores

def perform_grid_search(X_train, y_train, cv=5, param_grid=None):
    """
    Perform grid search for hyperparameter tuning.
    
    Topics: Algorithm Analysis (CV)
    
    Args:
        X_train: Training features
        y_train: Training labels
        cv: Number of cross-validation folds
        param_grid: Dictionary of parameters to search over, defaults to standard if None
    
    Returns:
        GridSearchCV: Fitted grid search object
    """
    print("\nPerforming grid search for Logistic Regression...")
    
    if param_grid is None:
        param_grid = {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear']  # For compatibility with both L1 and L2
        }
    
    grid_search = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        param_grid=param_grid,
        cv=cv,
        scoring='accuracy',
        n_jobs=-1
    )
    
    grid_search.fit(X_train, y_train)
    
    print(f"Best parameters: {grid_search.best_params_}")
    print(f"Best CV score: {grid_search.best_score_:.4f}")
    
    # Create a DataFrame of results
    results = pd.DataFrame(grid_search.cv_results_)
    
    # Extract relevant columns
    results_df = results[['param_C', 'param_penalty', 'mean_test_score', 'std_test_score', 'rank_test_score']]
    results_df = results_df.sort_values('rank_test_score')
    
    # Save results
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, 'grid_search_results.csv')
    results_df.to_csv(results_path, index=False)
    
    print(f"Grid search results saved to {results_path}")
    
    return grid_search

def analyze_tree_pruning(X_train, y_train, X_val, y_val, max_depth_values=None, n_estimators_values=None):
    """
    Analyze the effect of tree pruning in Random Forest.
    
    Topics: Algorithm Analysis (capacity, under/overfitting)
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        max_depth_values: List of max_depth values to try
        n_estimators_values: List of n_estimators values to try
    
    Returns:
        dict: Metrics from the analysis
    """
    print("\nAnalyzing tree pruning strategies in Random Forest...")
    
    if max_depth_values is None:
        max_depth_values = [None, 5, 10, 15, 20, 25, 30]
    
    if n_estimators_values is None:
        n_estimators_values = [100]  # Default to just using 100 estimators
    
    results = []
    best_val_accuracy = 0
    best_config = {}
    
    for n_estimators in n_estimators_values:
        for max_depth in max_depth_values:
            depth_str = str(max_depth) if max_depth is not None else "None"
            print(f"Training Random Forest with n_estimators={n_estimators}, max_depth={depth_str}...")
            
            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            model.fit(X_train, y_train)
            
            train_score = accuracy_score(y_train, model.predict(X_train))
            val_score = accuracy_score(y_val, model.predict(X_val))
            
            results.append({
                'n_estimators': n_estimators,
                'max_depth': max_depth,
                'train_accuracy': train_score,
                'val_accuracy': val_score
            })
            
            print(f"n_estimators={n_estimators}, max_depth={depth_str}: "
                  f"Train accuracy = {train_score:.4f}, Val accuracy = {val_score:.4f}")
            
            if val_score > best_val_accuracy:
                best_val_accuracy = val_score
                best_config = {
                    'n_estimators': n_estimators,
                    'max_depth': max_depth
                }
                # Save the best model
                os.makedirs(MODEL_DIR, exist_ok=True)
                best_model_path = os.path.join(MODEL_DIR, 'random_forest_best.pkl')
                save_pickle(model, best_model_path)
    
    # Create a DataFrame for easier analysis
    results_df = pd.DataFrame(results)
    
    # Save results to CSV
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(RESULTS_DIR, 'rf_pruning_results.csv')
    results_df.to_csv(results_path, index=False)
    
    # Plot results for different max_depth values with the best n_estimators
    best_n_estimators = best_config['n_estimators']
    best_results = results_df[results_df['n_estimators'] == best_n_estimators]
    
    # Convert None to a string for plotting
    best_results['max_depth_str'] = best_results['max_depth'].apply(lambda d: str(d) if d is not None else "None")
    
    # Sort by max_depth for plotting
    best_results = best_results.sort_values('max_depth', key=lambda x: x.fillna(1000))
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(best_results['max_depth_str'], best_results['train_accuracy'], 'bo-', label='Training Accuracy')
    plt.plot(best_results['max_depth_str'], best_results['val_accuracy'], 'ro-', label='Validation Accuracy')
    plt.xlabel('Max Depth')
    plt.ylabel('Accuracy')
    plt.title(f'Effect of Tree Pruning in Random Forest (n_estimators={best_n_estimators})')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    save_path = os.path.join(RESULTS_DIR, 'tree_pruning_analysis.png')
    plt.savefig(save_path)
    plt.close()
    
    print(f"Tree pruning analysis plot saved to {save_path}")
    print(f"Best Random Forest configuration: n_estimators={best_config['n_estimators']}, "
          f"max_depth={best_config['max_depth']}, validation accuracy={best_val_accuracy:.4f}")
    
    return {
        'results': results,
        'best_config': best_config,
        'best_val_accuracy': best_val_accuracy
    }

def main():
    """
    Main function to analyze models and their performance.
    """
    parser = argparse.ArgumentParser(description='Analyze models for SST-2 dataset')
    parser.add_argument('--feature-type', choices=['bow', 'tfidf'], default='tfidf',
                        help='Type of features to use')
    parser.add_argument('--cv', type=int, default=5,
                        help='Number of cross-validation folds')
    parser.add_argument('--expanded-search', action='store_true',
                        help='Use expanded hyperparameter search')
    args = parser.parse_args()
    
    # Load features
    features_path = os.path.join(PROCESSED_DATA_DIR, f'{args.feature_type}_features.pkl')
    features = load_pickle(features_path)
    
    X_train = features['X_train']
    y_train = features['y_train']
    X_val = features['X_val']
    y_val = features['y_val']
    
    # Ensure results directory exists
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Plot learning curves for Logistic Regression
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_save_path = os.path.join(RESULTS_DIR, 'lr_learning_curve.png')
    plot_learning_curve(
        lr_model, X_train, y_train, cv=args.cv,
        title="Learning Curve - Logistic Regression",
        save_path=lr_save_path
    )
    
    # Plot learning curves for Random Forest
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_save_path = os.path.join(RESULTS_DIR, 'rf_learning_curve.png')
    plot_learning_curve(
        rf_model, X_train, y_train, cv=args.cv,
        title="Learning Curve - Random Forest",
        save_path=rf_save_path
    )
    
    # Compare L1 vs L2 regularization
    compare_regularization(X_train, y_train, X_val, y_val, cv=args.cv)
    
    # Perform grid search with expanded search if requested
    if args.expanded_search:
        print("\nPerforming expanded grid search...")
        param_grid = {
            'C': np.logspace(-4, 4, 15),  # More values for C
            'penalty': ['l1', 'l2', 'elasticnet', None],  # More penalty options
            'solver': ['liblinear', 'saga'],  # More solvers
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]  # For elasticnet
        }
        grid_search = perform_grid_search(X_train, y_train, cv=args.cv, param_grid=param_grid)
    else:
        grid_search = perform_grid_search(X_train, y_train, cv=args.cv)
    
    # Analyze tree pruning strategies with expanded search if requested
    if args.expanded_search:
        print("\nPerforming expanded tree pruning analysis...")
        rf_metrics = analyze_tree_pruning(X_train, y_train, X_val, y_val, 
                                     max_depth_values=[None, 5, 10, 15, 20, 25, 30, 40, 50],
                                     n_estimators_values=[50, 100, 200, 300, 500])
    else:
        rf_metrics = analyze_tree_pruning(X_train, y_train, X_val, y_val)
    
    # Plot validation curve for C parameter in Logistic Regression
    lr_model = LogisticRegression(penalty='l2', max_iter=1000, random_state=42)
    C_values = np.logspace(-3, 3, 9) if args.expanded_search else np.logspace(-3, 3, 7)
    C_save_path = os.path.join(RESULTS_DIR, 'lr_C_validation_curve.png')
    _, train_scores, val_scores = plot_validation_curve(
        lr_model, X_train, y_train,
        param_name="C", param_range=C_values, cv=args.cv,
        title="Validation Curve - Logistic Regression (C parameter)",
        save_path=C_save_path
    )
    
    # Save metrics for performance gate validation
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Save Random Forest accuracy for performance gate
    rf_accuracy_data = {
        'value': rf_metrics.get('best_val_accuracy', 0),
        'random_forest': rf_metrics.get('best_val_accuracy', 0)
    }
    rf_accuracy_path = os.path.join(RESULTS_DIR, 'random_forest_accuracy.json')
    with open(rf_accuracy_path, 'w') as f:
        json.dump(rf_accuracy_data, f, indent=2)
    
    print(f"\nRandom Forest metrics saved to {rf_accuracy_path}")
    print(f"Best Random Forest accuracy: {rf_metrics.get('best_val_accuracy', 0):.4f}")
    
    # Save best logistic regression performance from validation curve
    # Check dimensions of val_scores and handle appropriately
    if val_scores.ndim > 1:
        # If val_scores is 2D, take mean along axis 1
        best_lr_idx = np.argmax(np.mean(val_scores, axis=1))
        best_lr_accuracy = np.mean(val_scores, axis=1)[best_lr_idx]
    else:
        # If val_scores is already 1D
        best_lr_idx = np.argmax(val_scores)
        best_lr_accuracy = val_scores[best_lr_idx]
    
    best_lr_c = C_values[best_lr_idx]
    
    lr_accuracy_data = {
        'value': best_lr_accuracy,
        'logistic_regression': best_lr_accuracy,
        'best_c': float(best_lr_c)
    }
    lr_accuracy_path = os.path.join(RESULTS_DIR, 'logistic_regression_accuracy.json')
    with open(lr_accuracy_path, 'w') as f:
        json.dump(lr_accuracy_data, f, indent=2)
    
    print(f"\nLogistic Regression metrics saved to {lr_accuracy_path}")
    print(f"Best Logistic Regression accuracy (C={best_lr_c:.4f}): {best_lr_accuracy:.4f}")
    
    print("\nModel analysis completed successfully!")

if __name__ == '__main__':
    main()

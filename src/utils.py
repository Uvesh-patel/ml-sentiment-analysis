"""
Utility functions for the SST-2 sentiment analysis project.
This module contains helper functions that are used across multiple scripts.
"""
import os
import sys
import json
import pickle
import importlib.util
import subprocess
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.metrics import silhouette_score, adjusted_rand_score

def ensure_dependencies():
    """
    Check if required dependencies are installed and install them if not.
    """
    dependencies = {
        'numpy': '1.24.0',
        'pandas': '1.5.3',  # Using an older, more stable version of pandas
        'scikit-learn': '1.3.0',
        'matplotlib': '3.7.0',
        'seaborn': '0.12.0',
        'tensorflow': '2.12.0',
        'nltk': '3.8.0',
        'datasets': '2.13.0',
        'transformers': '4.30.0',
        'wordcloud': '1.8.0'
    }
    missing_packages = []
    for package, version in dependencies.items():
        if importlib.util.find_spec(package) is None:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Installing missing dependencies: {', '.join(missing_packages)}")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing_packages)
            print("All dependencies installed successfully!")
        except subprocess.CalledProcessError:
            print("Error installing dependencies. Please run 'pip install -r requirements.txt' manually.")
            sys.exit(1)
    
    # Special case for NLTK resources, which are needed but not checked by importlib
    try:
        import nltk
        nltk_resources = ['punkt', 'stopwords', 'wordnet']
        for resource in nltk_resources:
            try:
                nltk.data.find(f'tokenizers/{resource}')
            except LookupError:
                print(f"Downloading NLTK resource: {resource}")
                nltk.download(resource, quiet=True)
    except ImportError:
        # NLTK will be installed above if missing
        pass

# Check for dependencies before proceeding
ensure_dependencies()

# Constants
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
RAW_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'raw')
PROCESSED_DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'src', 'models')

# Create a timestamped directory for results if running independently
# If running through run_pipeline.py, this will be overridden
TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
RUN_RESULTS_DIR = os.path.join(PROJECT_ROOT, 'runs', f'run_{TIMESTAMP}')

# Check if we're being run through run_pipeline.py by checking if the timestamped directory already exists
# Use the docs/results as fallback for backward compatibility
if 'RUN_RESULTS_DIR' in globals() or 'RUN_RESULTS_DIR' in locals():
    RESULTS_DIR = os.path.join(RUN_RESULTS_DIR, 'results')
else:
    # Allow override from environment variable if set by run_pipeline.py
    env_results_dir = os.environ.get('SST2_RESULTS_DIR')
    if env_results_dir:
        RESULTS_DIR = env_results_dir
    else:
        RESULTS_DIR = os.path.join(PROJECT_ROOT, 'docs', 'results')

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, RESULTS_DIR, os.path.dirname(RESULTS_DIR)]:
    os.makedirs(directory, exist_ok=True)

def save_pickle(obj, filepath):
    """Save an object to a pickle file."""
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Object saved to {filepath}")

def load_pickle(filepath):
    """Load an object from a pickle file."""
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    print(f"Object loaded from {filepath}")
    return obj

def save_dataframe(df, filepath, format='csv'):
    """Save a pandas DataFrame to a file."""
    if format.lower() == 'csv':
        df.to_csv(filepath, index=False)
    elif format.lower() == 'parquet':
        df.to_parquet(filepath, index=False)
    else:
        raise ValueError(f"Unsupported format: {format}")
    print(f"DataFrame saved to {filepath}")

def load_dataframe(filepath, format='csv'):
    """Load a pandas DataFrame from a file."""
    if format.lower() == 'csv':
        df = pd.read_csv(filepath)
    elif format.lower() == 'parquet':
        df = pd.read_parquet(filepath)
    else:
        raise ValueError(f"Unsupported format: {format}")
    print(f"DataFrame loaded from {filepath}")
    return df

def evaluate_classifier(model, X_test, y_test, model_name=None):
    """Evaluate a classifier on test data and return metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = None
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
    }
    
    if y_pred_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
    
    if model_name:
        print(f"Model: {model_name}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    if 'roc_auc' in metrics:
        print(f"ROC AUC: {metrics['roc_auc']:.4f}")
    
    return metrics

def plot_learning_curve(model, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5)):
    """Plot learning curves for a model."""
    from sklearn.model_selection import learning_curve
    
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes,
        scoring='accuracy', shuffle=True
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.title("Learning Curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.grid()
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    
    return plt

def evaluate_clustering(X, labels, true_labels=None):
    """Evaluate clustering results and return metrics."""
    metrics = {
        'silhouette': silhouette_score(X, labels)
    }
    
    if true_labels is not None:
        metrics['ari'] = adjusted_rand_score(true_labels, labels)
    
    print(f"Silhouette Score: {metrics['silhouette']:.4f}")
    if 'ari' in metrics:
        print(f"Adjusted Rand Index: {metrics['ari']:.4f}")
    
    return metrics


def get_model_registry_path():
    """Return path to model registry JSON file that tracks model history."""
    registry_dir = os.path.join(PROJECT_ROOT, "model_registry")
    os.makedirs(registry_dir, exist_ok=True)
    return os.path.join(registry_dir, "model_registry.json")


def update_model_registry(model_id, run_dir, model_path, metrics, epoch_count=None):
    """Update the model registry with info about a trained model.
    
    Args:
        model_id (str): Unique identifier for the model (e.g., 'lstm', 'cnn')
        run_dir (str): Directory path of the current run
        model_path (str): Path to the saved model file
        metrics (dict): Performance metrics of the model
        epoch_count (int): Number of epochs used in this training run
    """
    registry_path = get_model_registry_path()
    
    # Load existing registry or create new
    if os.path.exists(registry_path):
        with open(registry_path, 'r') as f:
            registry = json.load(f)
    else:
        registry = {}
    
    # Update or create entry for this model
    if model_id not in registry:
        registry[model_id] = {
            "training_history": [], 
            "current_path": model_path,
            "total_epochs": 0,
            "first_trained": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "baseline_metrics": metrics.copy()  # Save initial metrics as baseline
        }
    
    # Add new training record
    training_record = {
        "run_dir": run_dir,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "metrics": metrics,
        "model_path": model_path,
        "run_number": len(registry[model_id]["training_history"]) + 1
    }
    
    if epoch_count:
        training_record["epochs"] = epoch_count
        registry[model_id]["total_epochs"] = registry[model_id].get("total_epochs", 0) + epoch_count
    
    # Calculate improvement over baseline and previous run
    if len(registry[model_id]["training_history"]) > 0:
        prev_metrics = registry[model_id]["training_history"][-1]["metrics"]
        baseline_metrics = registry[model_id]["baseline_metrics"]
        
        # Calculate improvement percentages
        training_record["improvement_over_previous"] = {
            metric: ((metrics[metric] - prev_metrics[metric]) / prev_metrics[metric]) * 100 
            if prev_metrics[metric] > 0 else 0
            for metric in metrics if metric in prev_metrics
        }
        
        training_record["improvement_over_baseline"] = {
            metric: ((metrics[metric] - baseline_metrics[metric]) / baseline_metrics[metric]) * 100 
            if baseline_metrics[metric] > 0 else 0
            for metric in metrics if metric in baseline_metrics
        }
    
    registry[model_id]["training_history"].append(training_record)
    registry[model_id]["current_path"] = model_path
    registry[model_id]["last_updated"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Save updated registry
    with open(registry_path, 'w') as f:
        json.dump(registry, f, indent=2)
        
    return registry[model_id]


def create_performance_evolution_plot(model_id, registry_data=None):
    """Create plot showing how model performance evolves across training runs.
    
    Args:
        model_id (str): The model identifier (e.g., 'lstm', 'cnn')
        registry_data (dict): Registry data for this model, if None will be loaded from registry
        
    Returns:
        str: Path to the saved plot
    """
    if registry_data is None:
        registry_path = get_model_registry_path()
        if not os.path.exists(registry_path):
            print(f"No model registry found at {registry_path}")
            return None
            
        with open(registry_path, 'r') as f:
            registry = json.load(f)
            
        if model_id not in registry:
            print(f"No history found for model {model_id}")
            return None
            
        registry_data = registry[model_id]
    
    history = registry_data["training_history"]
    if len(history) <= 1:
        print(f"Need at least 2 training runs to create evolution plot, found {len(history)}")
        return None
    
    # Prepare data for plotting
    run_numbers = [entry["run_number"] for entry in history]
    accuracies = [entry["metrics"].get("accuracy", 0) for entry in history]
    f1_scores = [entry["metrics"].get("f1_score", 0) for entry in history]
    
    # Add ROC AUC if available in all entries
    has_roc_auc = all("roc_auc" in entry["metrics"] for entry in history)
    if has_roc_auc:
        roc_aucs = [entry["metrics"].get("roc_auc", 0) for entry in history]
    
    # Create the plot
    plt.figure(figsize=(12, 7))
    
    plt.plot(run_numbers, accuracies, marker='o', linewidth=2, label='Accuracy')
    plt.plot(run_numbers, f1_scores, marker='s', linewidth=2, label='F1 Score')
    if has_roc_auc:
        plt.plot(run_numbers, roc_aucs, marker='^', linewidth=2, label='ROC AUC')
    
    plt.title(f"Performance Evolution for {model_id.upper()} Model", fontsize=14)
    plt.xlabel("Training Run Number", fontsize=12)
    plt.ylabel("Metric Value", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(run_numbers)
    
    # Add improvement annotations
    for i in range(1, len(history)):
        if "improvement_over_previous" in history[i]:
            acc_improvement = history[i]["improvement_over_previous"].get("accuracy", 0)
            if abs(acc_improvement) > 0.5:  # Only annotate if change is significant
                plt.annotate(
                    f"{acc_improvement:.1f}%",
                    xy=(run_numbers[i], accuracies[i]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha='center',
                    fontsize=9,
                    color='green' if acc_improvement > 0 else 'red'
                )
    
    plt.legend()
    plt.tight_layout()
    
    # Save plot to model registry directory
    plot_path = os.path.join(os.path.dirname(get_model_registry_path()), 
                           f"{model_id}_evolution.png")
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path


def get_latest_run_dir():
    """Find the most recent run directory."""
    runs_dir = os.path.join(PROJECT_ROOT, "runs")
    if not os.path.exists(runs_dir):
        return None
        
    available_runs = sorted([d for d in os.listdir(runs_dir) if d.startswith("run_")], reverse=True)
    if not available_runs:
        return None
        
    return os.path.join(runs_dir, available_runs[0])


def load_previous_model(model_id):
    """Load the most recent version of a model from the registry.
    
    Args:
        model_id (str): Model identifier (e.g., 'lstm', 'cnn')
        
    Returns:
        tuple: (model_path, registry_entry) if found, else (None, None)
    """
    registry_path = get_model_registry_path()
    if not os.path.exists(registry_path):
        print("No model registry found.")
        return None, None
        
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    if model_id not in registry or not registry[model_id]["training_history"]:
        print(f"No existing model found for {model_id}")
        return None, None
    
    model_info = registry[model_id]
    model_path = model_info["current_path"]
    
    if not os.path.exists(model_path):
        print(f"Model file not found at {model_path}")
        return None, None
        
    return model_path, model_info


def create_model_evolution_summary(output_dir=None):
    """Create a summary of model evolution across all training runs.
    
    Args:
        output_dir (str): Directory to save the summary, defaults to current RESULTS_DIR
    
    Returns:
        str: Path to the saved summary file
    """
    registry_path = get_model_registry_path()
    if not os.path.exists(registry_path):
        print("No model registry found. Run training first.")
        return None
    
    with open(registry_path, 'r') as f:
        registry = json.load(f)
    
    if not registry:
        print("Model registry is empty. Run training first.")
        return None
    
    if output_dir is None:
        output_dir = RESULTS_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create summary DataFrame
    summary_rows = []
    
    for model_id, model_info in registry.items():
        if not model_info["training_history"]:
            continue
            
        for run in model_info["training_history"]:
            row = {
                "model_id": model_id,
                "run_number": run["run_number"],
                "date": run["date"],
                "run_dir": os.path.basename(run["run_dir"]),
            }
            
            # Add metrics
            for metric_name, metric_value in run["metrics"].items():
                row[f"{metric_name}"] = metric_value
            
            # Add improvement metrics if available
            if "improvement_over_baseline" in run:
                for metric_name, improvement in run["improvement_over_baseline"].items():
                    row[f"{metric_name}_vs_baseline_%"] = improvement
                    
            if "improvement_over_previous" in run:
                for metric_name, improvement in run["improvement_over_previous"].items():
                    row[f"{metric_name}_vs_previous_%"] = improvement
            
            summary_rows.append(row)
    
    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_rows)
    
    # Sort by model and run number
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["model_id", "run_number"])
        
        # Save to CSV
        summary_path = os.path.join(output_dir, "model_evolution_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        print(f"Model evolution summary saved to {summary_path}")
        return summary_path
    else:
        print("No training runs found in registry.")
        return None

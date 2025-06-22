"""
Feature engineering script for SST-2 sentiment analysis project.
This script creates features from preprocessed text using Bag-of-Words and TF-IDF.

Topics covered:
- Regression (feature vectors)
- Classification (feature vectors)
"""
import os
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
import argparse
from utils import PROCESSED_DATA_DIR, MODEL_DIR, load_dataframe, save_pickle

def create_bow_features(train_df, val_df, test_df, max_features=20000):
    """
    Create Bag-of-Words features using CountVectorizer.
    
    Args:
        train_df (pd.DataFrame): Training data with 'processed_text' column
        val_df (pd.DataFrame): Validation data with 'processed_text' column
        test_df (pd.DataFrame): Test data with 'processed_text' column
        max_features (int): Maximum number of features to extract
    
    Returns:
        dict: Dictionary containing the vectorizer and the feature matrices
    """
    print(f"Creating Bag-of-Words features (max_features={max_features})...")
    
    # Initialize the vectorizer
    vectorizer = CountVectorizer(max_features=max_features)
    
    # Fit on training data
    X_train_bow = vectorizer.fit_transform(train_df['processed_text'])
    
    # Transform validation and test data
    X_val_bow = vectorizer.transform(val_df['processed_text'])
    X_test_bow = vectorizer.transform(test_df['processed_text'])
    
    # Extract labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values if 'label' in test_df.columns else None
    
    print(f"Bag-of-Words features shape: {X_train_bow.shape}")
    
    return {
        'vectorizer': vectorizer,
        'X_train': X_train_bow,
        'X_val': X_val_bow,
        'X_test': X_test_bow,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': vectorizer.get_feature_names_out()
    }

def create_tfidf_features(train_df, val_df, test_df, max_features=20000):
    """
    Create TF-IDF features using TfidfVectorizer.
    
    Args:
        train_df (pd.DataFrame): Training data with 'processed_text' column
        val_df (pd.DataFrame): Validation data with 'processed_text' column
        test_df (pd.DataFrame): Test data with 'processed_text' column
        max_features (int): Maximum number of features to extract
    
    Returns:
        dict: Dictionary containing the vectorizer and the feature matrices
    """
    print(f"Creating TF-IDF features (max_features={max_features})...")
    
    # Initialize the vectorizer
    vectorizer = TfidfVectorizer(max_features=max_features)
    
    # Fit on training data
    X_train_tfidf = vectorizer.fit_transform(train_df['processed_text'])
    
    # Transform validation and test data
    X_val_tfidf = vectorizer.transform(val_df['processed_text'])
    X_test_tfidf = vectorizer.transform(test_df['processed_text'])
    
    # Extract labels
    y_train = train_df['label'].values
    y_val = val_df['label'].values
    y_test = test_df['label'].values if 'label' in test_df.columns else None
    
    print(f"TF-IDF features shape: {X_train_tfidf.shape}")
    
    return {
        'vectorizer': vectorizer,
        'X_train': X_train_tfidf,
        'X_val': X_val_tfidf,
        'X_test': X_test_tfidf,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'feature_names': vectorizer.get_feature_names_out()
    }

def create_feature_pipeline(feature_type='tfidf', max_features=20000):
    """
    Create a scikit-learn Pipeline for feature extraction.
    
    Args:
        feature_type (str): Type of features to extract ('bow' or 'tfidf')
        max_features (int): Maximum number of features to extract
    
    Returns:
        sklearn.pipeline.Pipeline: Feature extraction pipeline
    """
    if feature_type == 'bow':
        vectorizer = CountVectorizer(max_features=max_features)
    elif feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(max_features=max_features)
    else:
        raise ValueError(f"Invalid feature type: {feature_type}")
    
    # Create pipeline
    pipeline = Pipeline([
        ('vectorizer', vectorizer)
    ])
    
    return pipeline

def main():
    """
    Main function to create features from preprocessed SST-2 data.
    """
    parser = argparse.ArgumentParser(description='Create features for SST-2 dataset')
    parser.add_argument('--max-features', type=int, default=20000,
                        help='Maximum number of features to extract')
    args = parser.parse_args()
    
    # Load processed data
    train_path = os.path.join(PROCESSED_DATA_DIR, 'sst2_train_processed.csv')
    val_path = os.path.join(PROCESSED_DATA_DIR, 'sst2_val_processed.csv')
    test_path = os.path.join(PROCESSED_DATA_DIR, 'sst2_test_processed.csv')
    
    train_df = load_dataframe(train_path)
    val_df = load_dataframe(val_path)
    test_df = load_dataframe(test_path)
    
    # Handle NaN values in the processed text
    print("Checking for NaN values in processed text...")
    train_nan_count = train_df['processed_text'].isna().sum()
    val_nan_count = val_df['processed_text'].isna().sum()
    test_nan_count = test_df['processed_text'].isna().sum()
    
    print(f"NaN values: Train={train_nan_count}, Val={val_nan_count}, Test={test_nan_count}")
    
    # Fill NaN values with empty string
    train_df['processed_text'] = train_df['processed_text'].fillna('')
    val_df['processed_text'] = val_df['processed_text'].fillna('')
    test_df['processed_text'] = test_df['processed_text'].fillna('')
    
    # Create features
    bow_features = create_bow_features(
        train_df, val_df, test_df, max_features=args.max_features
    )
    
    tfidf_features = create_tfidf_features(
        train_df, val_df, test_df, max_features=args.max_features
    )
    
    # Save vectorizers and feature matrices
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save vectorizers
    bow_vectorizer_path = os.path.join(MODEL_DIR, 'bow_vectorizer.pkl')
    tfidf_vectorizer_path = os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl')
    
    save_pickle(bow_features['vectorizer'], bow_vectorizer_path)
    save_pickle(tfidf_features['vectorizer'], tfidf_vectorizer_path)
    
    # Save feature matrices
    bow_features_path = os.path.join(PROCESSED_DATA_DIR, 'bow_features.pkl')
    tfidf_features_path = os.path.join(PROCESSED_DATA_DIR, 'tfidf_features.pkl')
    
    save_pickle(bow_features, bow_features_path)
    save_pickle(tfidf_features, tfidf_features_path)
    
    # Create and save pipelines
    bow_pipeline = create_feature_pipeline('bow', args.max_features)
    tfidf_pipeline = create_feature_pipeline('tfidf', args.max_features)
    
    bow_pipeline_path = os.path.join(MODEL_DIR, 'bow_pipeline.pkl')
    tfidf_pipeline_path = os.path.join(MODEL_DIR, 'tfidf_pipeline.pkl')
    
    save_pickle(bow_pipeline, bow_pipeline_path)
    save_pickle(tfidf_pipeline, tfidf_pipeline_path)
    
    print("Feature engineering completed successfully!")
    
    # Print sample of top features
    print("\nTop 20 features in Bag-of-Words:")
    for feature in bow_features['feature_names'][:20]:
        print(feature, end=', ')
    
    print("\n\nTop 20 features in TF-IDF:")
    for feature in tfidf_features['feature_names'][:20]:
        print(feature, end=', ')
    print("\n")

if __name__ == '__main__':
    main()

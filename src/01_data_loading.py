"""
Data loading script for SST-2 sentiment analysis project.
This script downloads the SST-2 dataset from the Hugging Face datasets library
and stores it in the raw data directory.
"""
import os
import pandas as pd
from datasets import load_dataset
import argparse
from utils import RAW_DATA_DIR, save_dataframe

def download_sst2():
    """
    Download SST-2 dataset from Hugging Face datasets.
    Returns train, validation, and test splits as pandas DataFrames.
    """
    print("Loading SST-2 dataset from Hugging Face...")
    dataset = load_dataset("nyu-mll/glue", "sst2")
    
    # Convert to pandas DataFrames
    train_df = pd.DataFrame(dataset['train'])
    val_df = pd.DataFrame(dataset['validation'])
    test_df = pd.DataFrame(dataset['test'])
    
    return train_df, val_df, test_df

def analyze_dataset(train_df, val_df, test_df):
    """
    Analyze the dataset and print statistics.
    """
    print("\nDataset Statistics:")
    print(f"Train set: {len(train_df)} examples")
    print(f"Validation set: {len(val_df)} examples")
    print(f"Test set: {len(test_df)} examples")
    
    # Class balance
    train_class_counts = train_df['label'].value_counts()
    val_class_counts = val_df['label'].value_counts()
    
    print("\nClass Distribution (Train):")
    for label, count in train_class_counts.items():
        percentage = count / len(train_df) * 100
        print(f"Label {label}: {count} examples ({percentage:.2f}%)")
    
    print("\nClass Distribution (Validation):")
    for label, count in val_class_counts.items():
        percentage = count / len(val_df) * 100
        print(f"Label {label}: {count} examples ({percentage:.2f}%)")
    
    # Sentence length statistics
    train_df['sentence_length'] = train_df['sentence'].apply(lambda x: len(x.split()))
    
    print("\nSentence Length Statistics (Train):")
    print(f"Min length: {train_df['sentence_length'].min()} words")
    print(f"Max length: {train_df['sentence_length'].max()} words")
    print(f"Mean length: {train_df['sentence_length'].mean():.2f} words")
    print(f"Median length: {train_df['sentence_length'].median()} words")
    
    # Drop the temporary column
    train_df.drop('sentence_length', axis=1, inplace=True)

def save_datasets(train_df, val_df, test_df):
    """
    Save the datasets to CSV files in the raw data directory.
    """
    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    
    train_path = os.path.join(RAW_DATA_DIR, 'sst2_train.csv')
    val_path = os.path.join(RAW_DATA_DIR, 'sst2_val.csv')
    test_path = os.path.join(RAW_DATA_DIR, 'sst2_test.csv')
    
    save_dataframe(train_df, train_path)
    save_dataframe(val_df, val_path)
    save_dataframe(test_df, test_path)

def main():
    """
    Main function to download and save the SST-2 dataset.
    """
    parser = argparse.ArgumentParser(description='Download SST-2 dataset')
    parser.add_argument('--analyze', action='store_true', help='Analyze dataset statistics')
    args = parser.parse_args()
    
    print("Downloading SST-2 dataset...")
    train_df, val_df, test_df = download_sst2()
    
    if args.analyze:
        analyze_dataset(train_df, val_df, test_df)
    
    save_datasets(train_df, val_df, test_df)
    print("Dataset downloaded and saved successfully!")

if __name__ == '__main__':
    main()

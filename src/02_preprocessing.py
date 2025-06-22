"""
Text preprocessing script for SST-2 sentiment analysis project.
This script performs text cleaning and preprocessing on the SST-2 dataset.
"""
import os
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import argparse
from utils import RAW_DATA_DIR, PROCESSED_DATA_DIR, load_dataframe, save_dataframe

# Download NLTK resources
def download_nltk_resources():
    """Download required NLTK resources."""
    resources = ['punkt', 'stopwords', 'wordnet']
    for resource in resources:
        try:
            nltk.data.find(f'tokenizers/{resource}')
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=True)

def preprocess_text(text, remove_stopwords=True, lemmatize=False):
    """
    Preprocess the input text by performing the following steps:
    1. Convert to lowercase
    2. Remove special characters and punctuation
    3. Tokenize
    4. Remove stopwords (optional)
    5. Lemmatize (optional)
    
    Args:
        text (str): Input text to preprocess
        remove_stopwords (bool): Whether to remove stopwords
        lemmatize (bool): Whether to perform lemmatization
    
    Returns:
        str: Preprocessed text
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove special characters and punctuation
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Tokenize
    tokens = word_tokenize(text)
    
    # Remove stopwords
    if remove_stopwords:
        stop_words = set(stopwords.words('english'))
        tokens = [token for token in tokens if token not in stop_words]
    
    # Lemmatize
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    # Join tokens back into a string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

def preprocess_dataset(df, remove_stopwords=True, lemmatize=False):
    """
    Apply preprocessing to the entire dataset.
    
    Args:
        df (pd.DataFrame): Input DataFrame containing a 'sentence' column
        remove_stopwords (bool): Whether to remove stopwords
        lemmatize (bool): Whether to perform lemmatization
    
    Returns:
        pd.DataFrame: DataFrame with preprocessed text in a new 'processed_text' column
    """
    print(f"Preprocessing dataset with {len(df)} examples...")
    print(f"Options: remove_stopwords={remove_stopwords}, lemmatize={lemmatize}")
    
    # Create a copy of the DataFrame to avoid modifying the original
    processed_df = df.copy()
    
    # Apply preprocessing to each sentence
    processed_df['processed_text'] = processed_df['sentence'].apply(
        lambda x: preprocess_text(x, remove_stopwords, lemmatize)
    )
    
    return processed_df

def main():
    """
    Main function to preprocess the SST-2 dataset.
    """
    parser = argparse.ArgumentParser(description='Preprocess SST-2 dataset')
    parser.add_argument('--no-stopwords', action='store_false', dest='remove_stopwords',
                        help='Do not remove stopwords')
    parser.add_argument('--lemmatize', action='store_true', help='Perform lemmatization')
    args = parser.parse_args()
    
    # Download NLTK resources
    download_nltk_resources()
    
    # Load raw data
    train_path = os.path.join(RAW_DATA_DIR, 'sst2_train.csv')
    val_path = os.path.join(RAW_DATA_DIR, 'sst2_val.csv')
    test_path = os.path.join(RAW_DATA_DIR, 'sst2_test.csv')
    
    train_df = load_dataframe(train_path)
    val_df = load_dataframe(val_path)
    test_df = load_dataframe(test_path)
    
    # Apply preprocessing
    processed_train_df = preprocess_dataset(
        train_df, args.remove_stopwords, args.lemmatize
    )
    processed_val_df = preprocess_dataset(
        val_df, args.remove_stopwords, args.lemmatize
    )
    processed_test_df = preprocess_dataset(
        test_df, args.remove_stopwords, args.lemmatize
    )
    
    # Save processed data
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    processed_train_path = os.path.join(PROCESSED_DATA_DIR, 'sst2_train_processed.csv')
    processed_val_path = os.path.join(PROCESSED_DATA_DIR, 'sst2_val_processed.csv')
    processed_test_path = os.path.join(PROCESSED_DATA_DIR, 'sst2_test_processed.csv')
    
    save_dataframe(processed_train_df, processed_train_path)
    save_dataframe(processed_val_df, processed_val_path)
    save_dataframe(processed_test_df, processed_test_path)
    
    print("Preprocessing completed successfully!")
    
    # Sample output
    print("\nSample of preprocessed data:")
    for i in range(min(5, len(processed_train_df))):
        original = processed_train_df.iloc[i]['sentence']
        processed = processed_train_df.iloc[i]['processed_text']
        print(f"\nOriginal: {original}")
        print(f"Processed: {processed}")

if __name__ == '__main__':
    main()

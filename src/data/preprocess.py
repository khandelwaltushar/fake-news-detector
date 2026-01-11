"""
Data preprocessing script for fake news detection dataset.

This script processes raw CSV files (True.csv and Fake.csv) and creates
cleaned datasets ready for training with train/val/test splits.
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path


def remove_publisher_place(text: str) -> str:
    """
    Remove publisher and place of publication from the beginning of text.
    Pattern: "PLACE (Publisher) - " or "PLACE/PLACE (Publisher) - "
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Regex pattern to match: PLACE (Publisher) - 
    # Examples: "WASHINGTON (Reuters) - ", "SEATTLE/WASHINGTON (Reuters) - "
    pattern = r'^[A-Z][A-Z\s,/-]*\s*\([^)]+\)\s*-+\s*'
    
    # Remove the pattern and strip whitespace
    cleaned_text = re.sub(pattern, '', text).strip()
    
    return cleaned_text


def clean_content(text: str) -> str:
    """
    Clean content text by:
    1. Lowercasing
    2. Removing HTML tags
    3. Normalizing whitespace
    
    Args:
        text: Input text string
        
    Returns:
        Cleaned text string
    """
    if not isinstance(text, str):
        text = str(text)
    
    # 1. Lowercase
    text = text.lower()
    
    # 2. Remove HTML tags
    html_pattern = r'<[^>]+>'
    text = re.sub(html_pattern, '', text)
    
    # 3. Normalize whitespace (replace multiple whitespace with single space)
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text


def count_tokens(text: str) -> int:
    """
    Count tokens in text (using whitespace splitting as approximation).
    
    Args:
        text: Input text string
        
    Returns:
        Number of tokens
    """
    if not isinstance(text, str):
        text = str(text)
    return len(text.split())


def cap_tokens(text: str, max_tokens: int = 1000) -> str:
    """
    Cap text to maximum number of tokens.
    
    Args:
        text: Input text string
        max_tokens: Maximum number of tokens to keep
        
    Returns:
        Text capped at max_tokens
    """
    if not isinstance(text, str):
        text = str(text)
    
    tokens = text.split()
    capped_tokens = tokens[:max_tokens]
    
    return ' '.join(capped_tokens)


def preprocess_data(
    true_csv_path: str,
    fake_csv_path: str,
    output_dir: str,
    random_seed: int = 42,
    max_tokens: int = 1000,
    min_chars: int = 50
):
    """
    Preprocess fake news detection dataset.
    
    Args:
        true_csv_path: Path to True.csv file
        fake_csv_path: Path to Fake.csv file
        output_dir: Directory to save processed datasets
        random_seed: Random seed for reproducibility
        max_tokens: Maximum number of tokens per sample
        min_chars: Minimum character length threshold (samples with <= this many characters are removed)
    
    Returns:
        None (saves train.csv, val.csv, test.csv to output_dir)
    """
    print("=" * 60)
    print("DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load CSVs
    print("\n[1/8] Loading CSV files...")
    true_df = pd.read_csv(true_csv_path)
    fake_df = pd.read_csv(fake_csv_path)
    print(f"  ✓ Loaded True.csv: {true_df.shape}")
    print(f"  ✓ Loaded Fake.csv: {fake_df.shape}")
    
    # Step 2: Add labels
    print("\n[2/8] Adding labels...")
    true_df['label'] = 0  # REAL
    fake_df['label'] = 1  # FAKE
    print(f"  ✓ True dataset: label = 0 (REAL)")
    print(f"  ✓ Fake dataset: label = 1 (FAKE)")
    
    # Step 3: Remove publisher and place of publication from text column
    print("\n[3/8] Removing publisher/place pattern from text column...")
    true_before = true_df['text'].astype(str).str.contains(
        r'^[A-Z][A-Z\s,/-]*\s*\([^)]+\)\s*-+\s*', regex=True, na=False
    ).sum()
    fake_before = fake_df['text'].astype(str).str.contains(
        r'^[A-Z][A-Z\s,/-]*\s*\([^)]+\)\s*-+\s*', regex=True, na=False
    ).sum()
    
    true_df['text'] = true_df['text'].apply(remove_publisher_place)
    fake_df['text'] = fake_df['text'].apply(remove_publisher_place)
    
    true_after = true_df['text'].astype(str).str.contains(
        r'^[A-Z][A-Z\s,/-]*\s*\([^)]+\)\s*-+\s*', regex=True, na=False
    ).sum()
    fake_after = fake_df['text'].astype(str).str.contains(
        r'^[A-Z][A-Z\s,/-]*\s*\([^)]+\)\s*-+\s*', regex=True, na=False
    ).sum()
    
    print(f"  ✓ True dataset: {true_before} → {true_after} texts with pattern")
    print(f"  ✓ Fake dataset: {fake_before} → {fake_after} texts with pattern")
    
    # Step 4: Deduplicate articles
    print("\n[4/8] Deduplicating articles...")
    true_before_dedup = len(true_df)
    fake_before_dedup = len(fake_df)
    
    true_df = true_df.drop_duplicates(subset=['title', 'text'], keep='first').reset_index(drop=True)
    fake_df = fake_df.drop_duplicates(subset=['title', 'text'], keep='first').reset_index(drop=True)
    
    true_after_dedup = len(true_df)
    fake_after_dedup = len(fake_df)
    
    print(f"  ✓ True dataset: {true_before_dedup} → {true_after_dedup} samples ({true_before_dedup - true_after_dedup} duplicates removed)")
    print(f"  ✓ Fake dataset: {fake_before_dedup} → {fake_after_dedup} samples ({fake_before_dedup - fake_after_dedup} duplicates removed)")
    
    # Step 5: Concatenate title and text to create content column
    print("\n[5/8] Creating content column (title + text)...")
    true_df['content'] = true_df['title'].astype(str) + ' ' + true_df['text'].astype(str)
    fake_df['content'] = fake_df['title'].astype(str) + ' ' + fake_df['text'].astype(str)
    print(f"  ✓ Content column created for both datasets")
    
    # Step 6: Clean content column
    print("\n[6/8] Cleaning content column...")
    print("  - Lowercasing")
    print("  - Removing HTML tags")
    print("  - Normalizing whitespace")
    print("  - Capping at {} tokens".format(max_tokens))
    
    true_df['content'] = true_df['content'].apply(clean_content)
    fake_df['content'] = fake_df['content'].apply(clean_content)
    
    # Cap at max_tokens
    true_df['content'] = true_df['content'].apply(lambda x: cap_tokens(x, max_tokens))
    fake_df['content'] = fake_df['content'].apply(lambda x: cap_tokens(x, max_tokens))
    
    # Remove extremely short samples (character length <= min_chars)
    true_before_filter = len(true_df)
    fake_before_filter = len(fake_df)
    
    # Filter by character length > min_chars (matching notebook behavior)
    true_df = true_df[true_df['content'].str.len() > min_chars].reset_index(drop=True)
    fake_df = fake_df[fake_df['content'].str.len() > min_chars].reset_index(drop=True)
    
    true_after_filter = len(true_df)
    fake_after_filter = len(fake_df)
    
    print(f"  ✓ True dataset: {true_before_filter} → {true_after_filter} samples (removed {true_before_filter - true_after_filter} with <= {min_chars} characters)")
    print(f"  ✓ Fake dataset: {fake_before_filter} → {fake_after_filter} samples (removed {fake_before_filter - fake_after_filter} with <= {min_chars} characters)")
    
    # Step 7: Combine datasets and create final dataframe
    print("\n[7/8] Combining datasets...")
    # Select only the columns we need: text (content) and label
    true_final = true_df[['content', 'label']].copy()
    true_final.rename(columns={'content': 'text'}, inplace=True)
    
    fake_final = fake_df[['content', 'label']].copy()
    fake_final.rename(columns={'content': 'text'}, inplace=True)
    
    # Combine
    combined_df = pd.concat([true_final, fake_final], ignore_index=True)
    
    # Shuffle
    combined_df = combined_df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    
    print(f"  ✓ Combined dataset: {len(combined_df)} samples")
    print(f"    - REAL (label=0): {(combined_df['label'] == 0).sum()} samples")
    print(f"    - FAKE (label=1): {(combined_df['label'] == 1).sum()} samples")
    
    # Step 8: Stratified split (80% train, 10% val, 10% test)
    print("\n[8/8] Creating stratified train/val/test splits...")
    
    # First split: 80% train, 20% temp (which will become 10% val + 10% test)
    X = combined_df[['text']]
    y = combined_df['label']
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
        random_state=random_seed
    )
    
    # Second split: split temp into 50% val and 50% test (which is 10% and 10% of original)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=random_seed
    )
    
    # Create dataframes
    train_df = pd.DataFrame({'text': X_train['text'].values, 'label': y_train.values})
    val_df = pd.DataFrame({'text': X_val['text'].values, 'label': y_val.values})
    test_df = pd.DataFrame({'text': X_test['text'].values, 'label': y_test.values})
    
    print(f"  ✓ Train set: {len(train_df)} samples ({len(train_df)/len(combined_df)*100:.1f}%)")
    print(f"    - REAL: {(train_df['label'] == 0).sum()}, FAKE: {(train_df['label'] == 1).sum()}")
    print(f"  ✓ Validation set: {len(val_df)} samples ({len(val_df)/len(combined_df)*100:.1f}%)")
    print(f"    - REAL: {(val_df['label'] == 0).sum()}, FAKE: {(val_df['label'] == 1).sum()}")
    print(f"  ✓ Test set: {len(test_df)} samples ({len(test_df)/len(combined_df)*100:.1f}%)")
    print(f"    - REAL: {(test_df['label'] == 0).sum()}, FAKE: {(test_df['label'] == 1).sum()}")
    
    # Step 9: Save outputs
    print("\n[9/9] Saving processed datasets...")
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    train_path = output_path / 'train.csv'
    val_path = output_path / 'val.csv'
    test_path = output_path / 'test.csv'
    
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    
    print(f"  ✓ Saved train.csv: {train_path}")
    print(f"  ✓ Saved val.csv: {val_path}")
    print(f"  ✓ Saved test.csv: {test_path}")
    
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE!")
    print("=" * 60)
    
    return train_df, val_df, test_df


def main():
    """Main function to run preprocessing pipeline."""
    # Define paths
    script_dir = Path(__file__).parent.parent.parent
    data_raw_dir = script_dir / 'data' / 'raw'
    data_processed_dir = script_dir / 'data' / 'processed'
    
    true_csv_path = data_raw_dir / 'True.csv'
    fake_csv_path = data_raw_dir / 'Fake.csv'
    
    # Run preprocessing
    preprocess_data(
        true_csv_path=str(true_csv_path),
        fake_csv_path=str(fake_csv_path),
        output_dir=str(data_processed_dir),
        random_seed=42,
        max_tokens=1000,
        min_chars=50
    )


if __name__ == '__main__':
    main()

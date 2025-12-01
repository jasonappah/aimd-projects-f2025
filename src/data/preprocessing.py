"""Data preprocessing utilities for glucose time-series data."""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split


def load_data(csv_path: str) -> pd.DataFrame:
    """
    Load glucose time-series data from CSV.
    
    Args:
        csv_path: Path to CSV file
    
    Returns:
        DataFrame with parsed timestamps
    """
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(['id', 'timestamp']).reset_index(drop=True)
    return df


def handle_missing_values(df: pd.DataFrame, 
                         forward_fill: bool = True,
                         add_missing_indicators: bool = True) -> pd.DataFrame:
    """
    Handle missing values in the dataset.
    
    Args:
        df: Input DataFrame
        forward_fill: Whether to forward-fill missing values
        add_missing_indicators: Whether to add binary indicators for missing values
    
    Returns:
        DataFrame with handled missing values
    """
    df = df.copy()
    
    # Identify numeric columns (excluding id, hour, and binary flags)
    numeric_cols = ['glucose_mg_dL', 'carbs_g', 'insulin_units', 
                    'exercise_level', 'stress_level', 'age']
    
    # Forward fill missing values per person
    if forward_fill:
        for col in numeric_cols:
            if col in df.columns:
                df[col] = df.groupby('id')[col].ffill()
                # Fill remaining NaN with 0 or median
                if col == 'glucose_mg_dL':
                    df[col] = df[col].fillna(df.groupby('id')[col].transform('median'))
                else:
                    df[col] = df[col].fillna(0.0)
    
    # Add missing indicators
    if add_missing_indicators:
        for col in numeric_cols:
            if col in df.columns:
                df[f'{col}_missing'] = df[col].isna().astype(int)
                # Fill any remaining NaN after indicator creation
                if df[col].isna().any():
                    if col == 'glucose_mg_dL':
                        df[col] = df[col].fillna(df.groupby('id')[col].transform('median'))
                    else:
                        df[col] = df[col].fillna(0.0)
    
    # Handle activity_event (categorical)
    if 'activity_event' in df.columns:
        df['activity_event'] = df['activity_event'].fillna('none')
    
    return df


def create_person_splits(df: pd.DataFrame, 
                        test_size: float = 0.2,
                        val_size: float = 0.1,
                        random_state: int = 42) -> Tuple[List[int], List[int], List[int]]:
    """
    Create train/val/test splits by person ID (prevent data leakage).
    
    Args:
        df: Input DataFrame
        test_size: Proportion of people for test set
        val_size: Proportion of people for validation set (from remaining after test)
        random_state: Random seed
    
    Returns:
        Tuple of (train_ids, val_ids, test_ids) lists
    """
    unique_ids = df['id'].unique()
    np.random.seed(random_state)
    np.random.shuffle(unique_ids)
    
    n_total = len(unique_ids)
    n_test = int(n_total * test_size)
    n_val = int((n_total - n_test) * val_size)
    
    test_ids = unique_ids[:n_test].tolist()
    val_ids = unique_ids[n_test:n_test + n_val].tolist()
    train_ids = unique_ids[n_test + n_val:].tolist()
    
    return train_ids, val_ids, test_ids


def normalize_features(df: pd.DataFrame,
                      train_ids: List[int],
                      numeric_cols: List[str],
                      per_person: bool = False) -> Tuple[pd.DataFrame, Dict]:
    """
    Normalize numeric features.
    
    Args:
        df: Input DataFrame
        train_ids: List of person IDs in training set (for fit)
        numeric_cols: List of column names to normalize
        per_person: If True, normalize per person; if False, use global scaler
    
    Returns:
        Tuple of (normalized_df, scaler_dict)
    """
    df = df.copy()
    scalers = {}
    
    train_mask = df['id'].isin(train_ids)
    
    if per_person:
        # Normalize per person
        for person_id in df['id'].unique():
            person_mask = df['id'] == person_id
            person_train_mask = person_mask & train_mask
            
            scalers[person_id] = {}
            for col in numeric_cols:
                if col in df.columns:
                    scaler = StandardScaler()
                    train_data = df.loc[person_train_mask, col].values.reshape(-1, 1)
                    scaler.fit(train_data)
                    df.loc[person_mask, col] = scaler.transform(
                        df.loc[person_mask, col].values.reshape(-1, 1)
                    )
                    scalers[person_id][col] = scaler
    else:
        # Global normalization
        for col in numeric_cols:
            if col in df.columns:
                scaler = StandardScaler()
                train_data = df.loc[train_mask, col].values.reshape(-1, 1)
                scaler.fit(train_data)
                df[col] = scaler.transform(df[col].values.reshape(-1, 1))
                scalers[col] = scaler
    
    return df, scalers


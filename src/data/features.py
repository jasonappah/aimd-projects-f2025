"""Feature engineering for glucose time-series data."""

import pandas as pd
import numpy as np
from typing import List, Optional


def add_rolling_statistics(df: pd.DataFrame, 
                           windows: List[int] = [1, 3, 6],
                           cols: List[str] = ['glucose_mg_dL']) -> pd.DataFrame:
    """
    Add rolling statistics (mean, std, min) for specified columns.
    
    Args:
        df: Input DataFrame
        windows: List of window sizes in hours
        cols: Columns to compute rolling stats for
    
    Returns:
        DataFrame with added rolling statistics
    """
    df = df.copy()
    
    for col in cols:
        if col not in df.columns:
            continue
            
        for window in windows:
            # Group by person ID to compute rolling stats per person
            grouped = df.groupby('id')[col]
            
            # Rolling mean
            df[f'{col}_rolling_mean_{window}h'] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # Rolling std
            df[f'{col}_rolling_std_{window}h'] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).std().fillna(0)
            )
            
            # Rolling min
            df[f'{col}_rolling_min_{window}h'] = grouped.transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
    
    return df


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-based features (hour encoding, time-of-day).
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with added time features
    """
    df = df.copy()
    
    # Hour sin/cos encoding (24-hour cycle)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # Time of day categories (optional)
    df['is_night'] = ((df['hour'] >= 22) | (df['hour'] < 6)).astype(int)
    df['is_morning'] = ((df['hour'] >= 6) & (df['hour'] < 12)).astype(int)
    df['is_afternoon'] = ((df['hour'] >= 12) & (df['hour'] < 18)).astype(int)
    df['is_evening'] = ((df['hour'] >= 18) & (df['hour'] < 22)).astype(int)
    
    return df


def add_delta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add delta features (changes, ratios).
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with added delta features
    """
    df = df.copy()
    
    # Glucose change (1-hour delta)
    df['glucose_delta_1h'] = df.groupby('id')['glucose_mg_dL'].diff().fillna(0)
    
    # Insulin-to-carb ratio (when both present)
    df['insulin_to_carb_ratio'] = np.where(
        (df['carbs_g'] > 0) & (df['insulin_units'] > 0),
        df['insulin_units'] / (df['carbs_g'] + 1e-6),
        0.0
    )
    
    # Glucose rate of change (mg/dL per hour)
    df['glucose_rate'] = df.groupby('id')['glucose_delta_1h'].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    ).fillna(0)
    
    return df


def add_cumulative_features(df: pd.DataFrame, window_hours: int = 3) -> pd.DataFrame:
    """
    Add cumulative features over a time window.
    
    Args:
        df: Input DataFrame
        window_hours: Window size in hours
    
    Returns:
        DataFrame with added cumulative features
    """
    df = df.copy()
    
    # Cumulative insulin in last N hours
    df['insulin_cumulative_3h'] = df.groupby('id')['insulin_units'].transform(
        lambda x: x.rolling(window=window_hours, min_periods=1).sum()
    ).fillna(0)
    
    # Cumulative carbs in last N hours
    df['carbs_cumulative_3h'] = df.groupby('id')['carbs_g'].transform(
        lambda x: x.rolling(window=window_hours, min_periods=1).sum()
    ).fillna(0)
    
    return df


def add_time_since_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add time-since features (time since last meal, last insulin, etc.).
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with added time-since features
    """
    df = df.copy()
    
    # Time since last meal (activity_event with carbs)
    df['time_since_last_meal'] = 0
    for person_id in df['id'].unique():
        person_mask = df['id'] == person_id
        person_df = df[person_mask].copy()
        
        # Find meal events (non-empty activity_event with carbs > 0)
        meal_mask = (person_df['activity_event'].notna()) & \
                   (person_df['activity_event'] != 'none') & \
                   (person_df['carbs_g'] > 0)
        meal_indices = person_df[meal_mask].index
        
        if len(meal_indices) > 0:
            # Calculate hours since last meal
            for idx in person_df.index:
                if idx in meal_indices:
                    df.loc[idx, 'time_since_last_meal'] = 0
                else:
                    # Find most recent meal before this index
                    recent_meals = meal_indices[meal_indices < idx]
                    if len(recent_meals) > 0:
                        last_meal_idx = recent_meals[-1]
                        hours_since = (person_df.loc[idx, 'hour'] - 
                                      person_df.loc[last_meal_idx, 'hour'])
                        if hours_since < 0:
                            hours_since += 24  # Wrap around day
                        df.loc[idx, 'time_since_last_meal'] = hours_since
    
    # Time since last insulin
    df['time_since_last_insulin'] = 0
    for person_id in df['id'].unique():
        person_mask = df['id'] == person_id
        person_df = df[person_mask].copy()
        
        insulin_mask = person_df['insulin_units'] > 0
        insulin_indices = person_df[insulin_mask].index
        
        if len(insulin_indices) > 0:
            for idx in person_df.index:
                if idx in insulin_indices:
                    df.loc[idx, 'time_since_last_insulin'] = 0
                else:
                    recent_insulin = insulin_indices[insulin_indices < idx]
                    if len(recent_insulin) > 0:
                        last_insulin_idx = recent_insulin[-1]
                        hours_since = (person_df.loc[idx, 'hour'] - 
                                      person_df.loc[last_insulin_idx, 'hour'])
                        if hours_since < 0:
                            hours_since += 24
                        df.loc[idx, 'time_since_last_insulin'] = hours_since
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features (one-hot or label encoding).
    
    Args:
        df: Input DataFrame
    
    Returns:
        DataFrame with encoded categorical features
    """
    df = df.copy()
    
    # One-hot encode activity_event
    if 'activity_event' in df.columns:
        activity_dummies = pd.get_dummies(df['activity_event'], prefix='activity')
        df = pd.concat([df, activity_dummies], axis=1)
    
    # One-hot encode gender
    if 'gender' in df.columns:
        gender_dummies = pd.get_dummies(df['gender'], prefix='gender')
        df = pd.concat([df, gender_dummies], axis=1)
    
    # Encode exercise_level (if categorical, otherwise keep as numeric)
    if 'exercise_level' in df.columns:
        if df['exercise_level'].dtype == 'object':
            exercise_dummies = pd.get_dummies(df['exercise_level'], prefix='exercise')
            df = pd.concat([df, exercise_dummies], axis=1)
    
    return df


def create_all_features(df: pd.DataFrame,
                       rolling_windows: List[int] = [1, 3, 6],
                       rolling_cols: List[str] = ['glucose_mg_dL']) -> pd.DataFrame:
    """
    Create all engineered features.
    
    Args:
        df: Input DataFrame
        rolling_windows: Window sizes for rolling statistics
        rolling_cols: Columns for rolling statistics
    
    Returns:
        DataFrame with all features
    """
    df = add_rolling_statistics(df, windows=rolling_windows, cols=rolling_cols)
    df = add_time_features(df)
    df = add_delta_features(df)
    df = add_cumulative_features(df, window_hours=3)
    df = add_time_since_features(df)
    df = encode_categorical_features(df)
    
    return df


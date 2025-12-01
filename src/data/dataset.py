"""PyTorch Dataset classes for glucose time-series data."""

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from typing import List, Optional, Dict


class GlucoseTimeSeriesDataset(Dataset):
    """
    PyTorch Dataset for deep learning models (TCN, LSTM, Transformer).
    Creates sliding windows from time-series data.
    """
    
    def __init__(self,
                 df: pd.DataFrame,
                 window_size: int = 6,
                 feature_cols: Optional[List[str]] = None,
                 target_col: str = 'hypo_next_3_hours',
                 multitask: bool = False,
                 glucose_target_col: Optional[str] = 'glucose_mg_dL',
                 person_ids: Optional[List[int]] = None):
        """
        Initialize dataset.
        
        Args:
            df: Preprocessed DataFrame
            window_size: Number of hours in lookback window
            feature_cols: List of feature column names (if None, auto-detect)
            target_col: Target column name for classification
            multitask: If True, also return glucose regression target
            glucose_target_col: Column name for glucose regression target
            person_ids: Optional list of person IDs to include
        """
        self.df = df.copy()
        self.window_size = window_size
        self.target_col = target_col
        self.multitask = multitask
        self.glucose_target_col = glucose_target_col
        
        # Filter by person IDs if provided
        if person_ids is not None:
            self.df = self.df[self.df['id'].isin(person_ids)].copy()
        
        # Auto-detect feature columns if not provided
        if feature_cols is None:
            # Exclude metadata and target columns
            exclude_cols = ['id', 'timestamp', 'hour', target_col, 'hypo_flag',
                          'risk_score', 'prick_test', 'missing_measurement']
            if multitask and glucose_target_col:
                exclude_cols.append(glucose_target_col)
            
            # Get all numeric columns
            feature_cols = [col for col in self.df.columns 
                          if col not in exclude_cols and 
                          self.df[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
        
        self.feature_cols = feature_cols
        
        # Create sliding windows
        self.windows = self._create_windows()
    
    def _create_windows(self) -> List[Dict]:
        """
        Create sliding windows from the time-series data.
        
        Returns:
            List of window dictionaries
        """
        windows = []
        
        for person_id in self.df['id'].unique():
            person_df = self.df[self.df['id'] == person_id].sort_values('timestamp').reset_index(drop=True)
            
            # Create overlapping windows
            for i in range(len(person_df) - self.window_size + 1):
                window_df = person_df.iloc[i:i + self.window_size]
                
                # Extract features
                features = window_df[self.feature_cols].values.astype(np.float32)
                
                # Extract target (from the last row of the window)
                target = window_df.iloc[-1][self.target_col]
                
                # Extract glucose target for multitask (next hour after window)
                glucose_target = None
                if self.multitask and self.glucose_target_col:
                    if i + self.window_size < len(person_df):
                        glucose_target = person_df.iloc[i + self.window_size][self.glucose_target_col]
                    else:
                        glucose_target = window_df.iloc[-1][self.glucose_target_col]  # Use last value
                
                windows.append({
                    'features': features,
                    'target': target,
                    'glucose_target': glucose_target,
                    'person_id': person_id,
                    'window_start_idx': i,
                    'timestamp': window_df.iloc[-1]['timestamp']
                })
        
        return windows
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single window.
        
        Returns:
            Dictionary with 'features', 'target', and optionally 'glucose_target'
        """
        window = self.windows[idx]
        
        features = torch.FloatTensor(window['features'])
        target = torch.tensor(window['target'], dtype=torch.long)
        
        result = {
            'features': features,
            'target': target
        }
        
        if self.multitask and window['glucose_target'] is not None:
            result['glucose_target'] = torch.FloatTensor([window['glucose_target']])
        
        return result
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        return self.feature_cols


class TabularWindowDataset(Dataset):
    """
    PyTorch Dataset for baseline models (XGBoost, LightGBM, MLP).
    Flattens time windows into tabular format.
    """
    
    def __init__(self,
                 df: pd.DataFrame,
                 window_size: int = 6,
                 feature_cols: Optional[List[str]] = None,
                 target_col: str = 'hypo_next_3_hours',
                 person_ids: Optional[List[int]] = None,
                 flatten_strategy: str = 'last'):
        """
        Initialize tabular dataset.
        
        Args:
            df: Preprocessed DataFrame
            window_size: Number of hours in lookback window
            feature_cols: List of feature column names
            target_col: Target column name
            person_ids: Optional list of person IDs to include
            flatten_strategy: How to flatten window ('last', 'mean', 'concat')
        """
        self.df = df.copy()
        self.window_size = window_size
        self.target_col = target_col
        self.flatten_strategy = flatten_strategy
        
        if person_ids is not None:
            self.df = self.df[self.df['id'].isin(person_ids)].copy()
        
        if feature_cols is None:
            exclude_cols = ['id', 'timestamp', 'hour', target_col, 'hypo_flag',
                          'risk_score', 'prick_test', 'missing_measurement']
            feature_cols = [col for col in self.df.columns 
                          if col not in exclude_cols and 
                          self.df[col].dtype in [np.float64, np.int64, np.float32, np.int32]]
        
        self.feature_cols = feature_cols
        
        # Create flattened windows
        self.samples = self._create_tabular_samples()
    
    def _create_tabular_samples(self) -> List[Dict]:
        """Create tabular samples from windows."""
        samples = []
        
        for person_id in self.df['id'].unique():
            person_df = self.df[self.df['id'] == person_id].sort_values('timestamp').reset_index(drop=True)
            
            for i in range(len(person_df) - self.window_size + 1):
                window_df = person_df.iloc[i:i + self.window_size]
                
                # Flatten window based on strategy
                if self.flatten_strategy == 'last':
                    features = window_df.iloc[-1][self.feature_cols].values.astype(np.float32)
                elif self.flatten_strategy == 'mean':
                    features = window_df[self.feature_cols].mean().values.astype(np.float32)
                elif self.flatten_strategy == 'concat':
                    # Concatenate all time steps
                    features = window_df[self.feature_cols].values.flatten().astype(np.float32)
                else:
                    raise ValueError(f"Unknown flatten_strategy: {self.flatten_strategy}")
                
                target = window_df.iloc[-1][self.target_col]
                
                samples.append({
                    'features': features,
                    'target': target
                })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        return {
            'features': torch.FloatTensor(sample['features']),
            'target': torch.tensor(sample['target'], dtype=torch.long)
        }
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature column names."""
        if self.flatten_strategy == 'concat':
            # Return feature names with time step suffix
            return [f"{col}_t{i}" for i in range(self.window_size) for col in self.feature_cols]
        else:
            return self.feature_cols
    
    def get_numpy_features(self) -> np.ndarray:
        """Get all features as numpy array (for sklearn/XGBoost)."""
        return np.array([s['features'] for s in self.samples])
    
    def get_numpy_targets(self) -> np.ndarray:
        """Get all targets as numpy array (for sklearn/XGBoost)."""
        return np.array([s['target'] for s in self.samples])


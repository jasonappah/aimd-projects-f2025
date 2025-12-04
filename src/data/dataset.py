"""
Dataset class for blood glucose prediction.
Handles data loading, preprocessing, and train/val/test splitting.
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
import os


class BloodGlucoseDataset(Dataset):
    """
    PyTorch Dataset for blood glucose prediction.
    
    Handles:
    - Loading CSV data
    - Categorical encoding (one-hot for diabetes_type, exercise_intensity_level; 
      label encoding for exercise_name due to many unique values)
    - Numerical feature normalization
    - Train/val/test splitting
    """
    
    def __init__(self, csv_path, split='train', train_ratio=0.7, val_ratio=0.15, 
                 test_ratio=0.15, scaler_path=None, encoders_path=None, random_state=42):
        """
        Initialize the dataset.
        
        Args:
            csv_path: Path to the CSV file
            split: 'train', 'val', or 'test'
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            scaler_path: Path to save/load the scaler (optional)
            encoders_path: Path to save/load the encoders (optional)
            random_state: Random seed for reproducibility
        """
        self.split = split
        self.csv_path = csv_path
        self.random_state = random_state
        
        # Load and preprocess data
        self._load_and_preprocess_data(train_ratio, val_ratio, test_ratio, 
                                       scaler_path, encoders_path)
    
    def _load_and_preprocess_data(self, train_ratio, val_ratio, test_ratio,
                                  scaler_path, encoders_path):
        """Load CSV and perform preprocessing."""
        # Load data
        df = pd.read_csv(self.csv_path)
        
        # Drop event_id and patient_id (not features for prediction)
        df = df.drop(columns=['event_id', 'patient_id'], errors='ignore')
        
        # Separate target
        target_col = 'next_3hr_glucose'
        y = df[target_col].values.astype(np.float32)
        df = df.drop(columns=[target_col])
        
        # Identify categorical and numerical columns
        categorical_cols = ['diabetes_type', 'exercise_name', 'exercise_intensity_level']
        numerical_cols = [col for col in df.columns if col not in categorical_cols]
        
        # Handle missing values in categorical columns
        for col in categorical_cols:
            df[col] = df[col].fillna('unknown')
        
        # Handle missing values in numerical columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
        
        # Encode categorical features
        # Always fit encoders on full dataset for consistency
        encoders = None
        if encoders_path and os.path.exists(encoders_path):
            # Load existing encoders
            with open(encoders_path, 'rb') as f:
                encoders = pickle.load(f)
            diabetes_encoder = encoders['diabetes']
            exercise_name_encoder = encoders['exercise_name']
            intensity_encoder = encoders['intensity']
            
            # Transform using loaded encoders
            df['diabetes_type_encoded'] = diabetes_encoder.transform(df['diabetes_type'])
            df['exercise_name_encoded'] = exercise_name_encoder.transform(df['exercise_name'])
            df['exercise_intensity_level_encoded'] = intensity_encoder.transform(df['exercise_intensity_level'])
        else:
            # Create new encoders and fit on full dataset
            diabetes_encoder = LabelEncoder()
            exercise_name_encoder = LabelEncoder()
            intensity_encoder = LabelEncoder()
            
            # Fit encoders on full dataset
            df['diabetes_type_encoded'] = diabetes_encoder.fit_transform(df['diabetes_type'])
            df['exercise_name_encoded'] = exercise_name_encoder.fit_transform(df['exercise_name'])
            df['exercise_intensity_level_encoded'] = intensity_encoder.fit_transform(df['exercise_intensity_level'])
            
            # Create encoders dict
            encoders = {
                'diabetes': diabetes_encoder,
                'exercise_name': exercise_name_encoder,
                'intensity': intensity_encoder
            }
        
        # One-hot encode diabetes_type and exercise_intensity_level
        # Ensure consistent columns across splits
        diabetes_onehot = pd.get_dummies(df['diabetes_type'], prefix='diabetes', dtype=np.float32)
        intensity_onehot = pd.get_dummies(df['exercise_intensity_level'], prefix='intensity', dtype=np.float32)
        
        # If encoders exist and have column info, ensure consistency
        if encoders and 'diabetes_columns' in encoders and 'intensity_columns' in encoders:
            diabetes_columns = encoders['diabetes_columns']
            intensity_columns = encoders['intensity_columns']
            # Reindex to ensure all columns exist
            diabetes_onehot = diabetes_onehot.reindex(columns=diabetes_columns, fill_value=0.0)
            intensity_onehot = intensity_onehot.reindex(columns=intensity_columns, fill_value=0.0)
        else:
            # Save column structure for consistency (first time)
            if encoders_path:
                encoders['diabetes_columns'] = list(diabetes_onehot.columns)
                encoders['intensity_columns'] = list(intensity_onehot.columns)
                os.makedirs(os.path.dirname(encoders_path) if os.path.dirname(encoders_path) else '.', exist_ok=True)
                with open(encoders_path, 'wb') as f:
                    pickle.dump(encoders, f)
        
        # Combine features
        numerical_features = df[numerical_cols].values.astype(np.float32)
        exercise_name_encoded = df['exercise_name_encoded'].values.astype(np.float32).reshape(-1, 1)
        
        # Combine all features
        X = np.hstack([
            numerical_features,
            diabetes_onehot.values,
            intensity_onehot.values,
            exercise_name_encoded
        ])
        
        # Split data - use consistent random state for reproducible splits
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_ratio, random_state=self.random_state
        )
        
        val_size_adjusted = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, random_state=self.random_state
        )
        
        # Handle scaler - fit on training data only
        num_features_count = len(numerical_cols)
        if scaler_path and os.path.exists(scaler_path):
            # Load existing scaler
            with open(scaler_path, 'rb') as f:
                scaler = pickle.load(f)
        else:
            # Create and fit scaler on training data only
            scaler = StandardScaler()
            X_train_numerical = X_train[:, :num_features_count]
            scaler.fit(X_train_numerical)
            
            # Save scaler if path provided
            if scaler_path:
                os.makedirs(os.path.dirname(scaler_path) if os.path.dirname(scaler_path) else '.', exist_ok=True)
                with open(scaler_path, 'wb') as f:
                    pickle.dump(scaler, f)
        
        # Apply scaling to numerical features only
        if self.split == 'train':
            X_train[:, :num_features_count] = scaler.transform(X_train[:, :num_features_count])
            self.features = torch.tensor(X_train, dtype=torch.float32)
            self.targets = torch.tensor(y_train, dtype=torch.float32)
        elif self.split == 'val':
            X_val[:, :num_features_count] = scaler.transform(X_val[:, :num_features_count])
            self.features = torch.tensor(X_val, dtype=torch.float32)
            self.targets = torch.tensor(y_val, dtype=torch.float32)
        else:  # test
            X_test[:, :num_features_count] = scaler.transform(X_test[:, :num_features_count])
            self.features = torch.tensor(X_test, dtype=torch.float32)
            self.targets = torch.tensor(y_test, dtype=torch.float32)
        
        # Store feature count for model initialization
        self.num_features = self.features.shape[1]
        
        print(f"{self.split.capitalize()} set: {len(self.features)} samples, {self.num_features} features")
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]

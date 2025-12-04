"""
Main training script for blood glucose prediction model.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import time

from data import BloodGlucoseDataset
from models import BloodGlucoseModel
from utils import get_device


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device).unsqueeze(1)  # Shape: (batch_size, 1)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device).unsqueeze(1)
            
            outputs = model(features)
            loss = criterion(outputs, targets)
            
            total_loss += loss.item()
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    
    # Calculate metrics
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    r2 = r2_score(all_targets, all_predictions)
    
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, mae, rmse, r2


def evaluate_test(model, dataloader, device):
    """Evaluate on test set with detailed metrics."""
    model.eval()
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device)
            
            outputs = model(features)
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(targets.numpy())
    
    all_predictions = np.concatenate(all_predictions, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()
    
    mae = mean_absolute_error(all_targets, all_predictions)
    rmse = np.sqrt(mean_squared_error(all_targets, all_predictions))
    r2 = r2_score(all_targets, all_predictions)
    
    return mae, rmse, r2, all_predictions, all_targets


def main():
    parser = argparse.ArgumentParser(description='Train blood glucose prediction model')
    parser.add_argument('--data', type=str, default='raw_data/blood_glucose_dataset_100k_exercise_name.csv',
                        help='Path to CSV data file')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden-sizes', type=int, nargs='+', default=[256, 128, 64, 32],
                        help='Hidden layer sizes')
    parser.add_argument('--dropout', type=float, default=0.4, help='Dropout rate')
    parser.add_argument('--loss', type=str, default='mse', choices=['mse', 'huber'],
                        help='Loss function')
    parser.add_argument('--early-stopping', type=int, default=10,
                        help='Early stopping patience')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory to save model checkpoints')
    parser.add_argument('--scaler-path', type=str, default='checkpoints/scaler.pkl',
                        help='Path to save/load scaler')
    parser.add_argument('--encoders-path', type=str, default='checkpoints/encoders.pkl',
                        help='Path to save/load encoders')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--num-workers', type=int, default=4, help='Number of data loading workers')
    
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    print("=" * 60)
    print("Blood Glucose Prediction Model Training")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.lr}")
    print(f"Hidden sizes: {args.hidden_sizes}")
    print(f"Dropout rate: {args.dropout}")
    print(f"Loss function: {args.loss}")
    print("=" * 60)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = BloodGlucoseDataset(
        args.data, split='train',
        scaler_path=args.scaler_path,
        encoders_path=args.encoders_path,
        random_state=args.seed
    )
    val_dataset = BloodGlucoseDataset(
        args.data, split='val',
        scaler_path=args.scaler_path,
        encoders_path=args.encoders_path,
        random_state=args.seed
    )
    test_dataset = BloodGlucoseDataset(
        args.data, split='test',
        scaler_path=args.scaler_path,
        encoders_path=args.encoders_path,
        random_state=args.seed
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda')
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda')
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == 'cuda')
    )
    
    # Create model
    model = BloodGlucoseModel(
        input_size=train_dataset.num_features,
        hidden_sizes=args.hidden_sizes,
        dropout_rate=args.dropout
    ).to(device)
    
    print(f"\nModel created with {model.get_num_parameters():,} parameters")
    print(f"Input size: {train_dataset.num_features}")
    
    # Loss function
    if args.loss == 'mse':
        criterion = nn.MSELoss()
    else:  # huber
        criterion = nn.HuberLoss(delta=1.0)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    print("\nStarting training...")
    print("-" * 60)
    
    start_time = time.time()
    
    for epoch in range(args.epochs):
        epoch_start = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_mae, val_rmse, val_r2 = validate(model, val_loader, criterion, device)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            # Save best model
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_mae': val_mae,
                'val_rmse': val_rmse,
                'val_r2': val_r2,
                'model_config': {
                    'input_size': train_dataset.num_features,
                    'hidden_sizes': args.hidden_sizes,
                    'dropout_rate': args.dropout
                }
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'best_model.pth'))
        else:
            patience_counter += 1
        
        epoch_time = time.time() - epoch_start
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{args.epochs}] "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val MAE: {val_mae:.2f} | "
                  f"Val RMSE: {val_rmse:.2f} | "
                  f"Val R²: {val_r2:.4f} | "
                  f"Time: {epoch_time:.2f}s")
        
        # Early stopping
        if patience_counter >= args.early_stopping:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Training completed in {total_time/60:.2f} minutes")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    # Load best model and evaluate on test set
    print("\nLoading best model for test evaluation...")
    checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print("Evaluating on test set...")
    test_mae, test_rmse, test_r2, test_predictions, test_targets = evaluate_test(
        model, test_loader, device
    )
    
    print("\n" + "=" * 60)
    print("Test Set Results:")
    print("=" * 60)
    print(f"MAE:  {test_mae:.2f} mg/dL")
    print(f"RMSE: {test_rmse:.2f} mg/dL")
    print(f"R²:   {test_r2:.4f}")
    print("=" * 60)
    
    # Save final model
    final_model_path = os.path.join(args.checkpoint_dir, 'final_model.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': checkpoint['model_config'],
        'test_metrics': {
            'mae': test_mae,
            'rmse': test_rmse,
            'r2': test_r2
        }
    }, final_model_path)
    print(f"\nModel saved to {final_model_path}")


if __name__ == '__main__':
    main()


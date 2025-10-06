"""
Train models on SVDF-20 dataset
Self-contained training script
"""
import os
import sys
import argparse
import logging
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from pathlib import Path

from config import args
from data_loader import get_svdf20_dataloaders
from src.models.models import get_model
from utils import create_optimizer, seed_worker, set_seed, str_to_bool, reproducibility


def train_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target, _, _, _) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{100. * correct / total:.2f}%'
        })
    
    return total_loss / len(dataloader), 100. * correct / total

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target, _, _, _ in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy

def update_args_from_command_line():
    """Update args from command line arguments"""
    import sys
    # Parse command line arguments and update the global args object
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith('--'):
            key = arg[2:]  # Remove '--'
            if i + 1 < len(sys.argv) and not sys.argv[i + 1].startswith('--'):
                value = sys.argv[i + 1]
                # Convert value to appropriate type
                if hasattr(args, key):
                    current_value = getattr(args, key)
                    if isinstance(current_value, bool):
                        setattr(args, key, str_to_bool(value))
                    elif isinstance(current_value, int):
                        setattr(args, key, int(value))
                    elif isinstance(current_value, float):
                        setattr(args, key, float(value))
                    else:
                        setattr(args, key, value)

def get_gpu_info():
    """Get GPU information for current process"""
    if torch.cuda.is_available():
        gpu_id = torch.cuda.current_device()
        memory_allocated = torch.cuda.memory_allocated(gpu_id)
        memory_reserved = torch.cuda.memory_reserved(gpu_id)
        memory_free = torch.cuda.get_device_properties(gpu_id).total_memory - memory_allocated
    else:
        gpu_id = -1
        memory_allocated = memory_reserved = memory_free = 0
    
    gpu_info = {
        'gpu_id': gpu_id,
        'memory_free': memory_free,
        'memory_allocated': memory_allocated,
        'memory_reserved': memory_reserved
    }
    
    return gpu_id, gpu_info

def main():
    """Main training function for SVDF-20"""
    # Update args from command line arguments
    update_args_from_command_line()
    
    # Use max_files from command line, or default to 1200 for subset training
    if args.max_files is None:
        args.max_files = 1200  # Use subset for faster training
        print(f"Using subset training with {args.max_files} files")
    
    # Get GPU info for the current process (respects CUDA_VISIBLE_DEVICES)
    gpu_id, gpu_info = get_gpu_info()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'svdf20_training_{args.model_name}_gpu{gpu_id}.log'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info(f"Starting SVDF-20 training for {args.model_name}")
    logger.info("=" * 80)
    # Get the actual physical GPU ID from CUDA_VISIBLE_DEVICES
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
    physical_gpu_id = cuda_visible.split(',')[0] if cuda_visible else '0'
    logger.info(f" Using GPU: {physical_gpu_id} (CUDA_VISIBLE_DEVICES={cuda_visible})")
    logger.info(f" GPU Memory: {gpu_info['memory_free']/1024**3:.1f}GB free, {gpu_info['memory_allocated']/1024**3:.1f}GB used")
    logger.info(f"Dataset: {args.dataset_name}")
    logger.info(f"Model: {args.model_name}")
    # Get model-specific batch size
    from data_loader import get_batch_size_for_model
    actual_batch_size = get_batch_size_for_model(args.model_name, args.batch_size)
    logger.info(f"Batch size: {actual_batch_size} (adjusted for {args.model_name} model)")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Mixed precision: {args.mixed_precision}")
    logger.info(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info("=" * 80)
    
    # Update args for SVDF-20 (use configurable paths)
    if not hasattr(args, 'database_path') or args.database_path is None:
        args.database_path = os.getenv('SVDF_DATASET_PATH', './dataset')
    args.dataset_name = 'SVDF-20'
    args.train_language = 'Multilingual'  # 20 languages
    args.test_language = 'Multilingual'
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.test_dir, exist_ok=True)
    
    try:
        # Get dataloaders
        train_loader, dev_loader = get_svdf20_dataloaders(args)
        
        # Get model
        model = get_model(args.model_name)
        
        # Setup training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        # Setup logging
        log_dir = Path('logs') / f"{args.model_name}_{args.dataset_name}"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup optimizer and criterion
        optimizer = create_optimizer(model, args)
        criterion = nn.CrossEntropyLoss()
        
        # Setup tensorboard
        writer = SummaryWriter(log_dir)
        
        best_acc = 0.0
        best_epoch = 0
        
        logger.info(f"Starting training for {args.num_epochs} epochs")
        logger.info(f"Model: {args.model_name}")
        logger.info(f"Batch size: {args.batch_size}")
        logger.info(f"Learning rate: {args.lr}")
        
        for epoch in range(1, args.num_epochs + 1):
            # Training
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, epoch)
            
            # Validation
            val_loss, val_acc = validate_epoch(model, dev_loader, criterion, device)
            
            # Log metrics
            writer.add_scalar('Loss/Train', train_loss, epoch)
            writer.add_scalar('Loss/Val', val_loss, epoch)
            writer.add_scalar('Accuracy/Train', train_acc, epoch)
            writer.add_scalar('Accuracy/Val', val_acc, epoch)
            
            logger.info(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            logger.info(f"Epoch {epoch}: Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            
            # Save best model
            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                torch.save(model.state_dict(), log_dir / 'best_model.pt')
                logger.info(f"New best model saved with accuracy: {best_acc:.2f}%")
        
        # Save final model
        checkpoint_path = f"{args.output_dir}/{args.model_name}_final.pt"
        torch.save(model.state_dict(), checkpoint_path)
        
        logger.info("=" * 80)
        logger.info(f" {args.model_name} training completed successfully!")
        logger.info(f"Best accuracy: {best_acc:.2f}% at epoch {best_epoch}")
        logger.info(f"Model saved at: {checkpoint_path}")
        logger.info("=" * 80)
        
        writer.close()
        return checkpoint_path
        
    except Exception as e:
        logger.error(f" {args.model_name} training failed: {e}")
        raise e

if __name__ == "__main__":
    main()

"""
SVDF-20 Dataset Dataloader for Audio Spoofing Detection
Adapted from SingFake dataloader for SVDF-20 dataset
"""
import soundfile as sf
import os
import torch
import numpy as np
import torch.utils.data as data
from torch.utils.data import DataLoader, Dataset
import librosa
from torch import Tensor
import pandas as pd
import glob
from typing import Tuple, List
import argparse
from pathlib import Path

torch.set_default_tensor_type(torch.FloatTensor)

SampleType = Tuple[Tensor, int, str, str, str]

def pad(x, max_len=64600):
    """Pad or truncate audio to fixed length"""
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    # need to pad
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

def get_audio_length_for_model(model_name):
    """Get appropriate audio length for different models"""
    if model_name == "Whisper":
        return 480000  # 30 seconds at 16kHz for Whisper
    else:
        return 64600   # Default length for other models

def get_batch_size_for_model(model_name, default_batch_size):
    """Get appropriate batch size for different models"""
    if model_name == "Whisper":
        return 4  # Reduced batch size for Whisper due to 30s audio
    else:
        return default_batch_size

class SVDF20Dataset(Dataset):
    """SVDF-20 Dataset for Audio Spoofing Detection"""
    
    def __init__(self, args, mode='train', split='Training'):
        """
        Args:
            args: Arguments from config
            mode: 'train', 'dev', 'test'
            split: 'Training', 'Validation', 'T01', 'T02', 'T03', 'T04'
        """
        self.args = args
        self.mode = mode
        self.split = split
        
        # Set paths based on mode and split
        if mode == 'train':
            if split == 'Validation':
                self.root = os.path.join(args.database_path, 'final_splits', 'Validation')
            else:
                self.root = os.path.join(args.database_path, 'final_splits', 'Training')
        elif mode == 'dev':
            self.root = os.path.join(args.database_path, 'final_splits', 'Validation')
        else:  # test
            self.root = os.path.join(args.database_path, 'final_splits', split)
        
        print(f"Loading SVDF-20 dataset - Mode: {mode}, Split: {split}")
        print(f"Root directory: {self.root}")
        
        # Load audio files from vocals directory (more efficient approach)
        vocals_dir = os.path.join(self.root, 'vocals')
        if os.path.exists(vocals_dir):
            # Use os.listdir instead of glob for better performance
            self.audio_files = []
            for filename in os.listdir(vocals_dir):
                if filename.endswith('.flac'):
                    self.audio_files.append(os.path.join(vocals_dir, filename))
        else:
            # Try mixtures if vocals not found
            mixtures_dir = os.path.join(self.root, 'mixtures')
            self.audio_files = []
            for filename in os.listdir(mixtures_dir):
                if filename.endswith('.flac'):
                    self.audio_files.append(os.path.join(mixtures_dir, filename))
            print(f"Using mixtures instead of vocals")
        
        print(f"Total audio files found: {len(self.audio_files)}")
        
        # Load labels from log files
        self.labels = self._load_labels()
        
        # Filter files based on available labels (optimized approach)
        print("Filtering files based on available labels...")
        # Create a set of valid file IDs for faster lookup
        valid_file_ids = set(self.labels.keys())
        filtered_files = []
        
        for f in self.audio_files:
            filename = os.path.basename(f)
            # Extract video_id directly (faster than _get_file_id)
            parts = filename.split('_')
            if len(parts) >= 3:
                video_id = parts[1]
                # Check if this video_id has a label
                if video_id in valid_file_ids:
                    filtered_files.append(f)
                # Also check if the full filename is in labels
                elif filename.split('.')[0] in valid_file_ids:
                    filtered_files.append(f)
        
        self.audio_files = filtered_files
        print(f"Audio files with labels: {len(self.audio_files)}")
        
        # For testing: limit number of files AFTER filtering
        if hasattr(self.args, 'max_files') and self.args.max_files:
            max_files = min(self.args.max_files, len(self.audio_files))
            self.audio_files = self.audio_files[:max_files]
            print(f"Limited to {max_files} files for testing")
        
        # Shuffle for training
        if mode == 'train':
            np.random.shuffle(self.audio_files)
    
    def _load_labels(self):
        """Load labels from metadata cache for better performance"""
        labels = {}
        self.filename_mapping = {}  # Map processed filenames to original filenames
        
        # Try to load from metadata cache first (much faster)
        metadata_cache_path = os.path.join(self.args.database_path, 'metadata_cache.json')
        if os.path.exists(metadata_cache_path):
            print(f"Loading labels from metadata cache: {metadata_cache_path}")
            import json
            with open(metadata_cache_path, 'r') as f:
                metadata = json.load(f)
            
            for filename, info in metadata.items():
                # Extract video_id from filename for mapping
                if '_' in filename:
                    video_id = filename.split('_')[1] if len(filename.split('_')) > 1 else filename
                    labels[video_id] = {
                        'label': 1 if info.get('is_spoof', False) else 0,
                        'singer': info.get('singer', 'Unknown'),
                        'language': info.get('language', 'Unknown'),
                        'model': info.get('model', 'Unknown')
                    }
            print(f"Loaded {len(labels)} labels from metadata cache")
            return labels
        
        # Fallback to log files if cache not available
        print("Metadata cache not found, falling back to log files...")
        logs_dir = os.path.join(self.args.database_path, 'logs')
        if os.path.exists(logs_dir):
            log_files = glob.glob(os.path.join(logs_dir, '*.log'))
            
            for log_file in log_files:
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) >= 5:
                            original_filename = lines[0].strip()
                            title = lines[1].strip()
                            url = lines[2].strip()
                            singer = lines[3].strip()
                            label_text = lines[4].strip()
                            
                            # Create file ID from original filename
                            file_id = original_filename
                            
                            # Handle different label formats
                            if label_text.lower() == 'bonafide':
                                label = 0
                            elif label_text.lower() == 'spoof':
                                label = 1
                            elif label_text.lower() == 't01':  # Bonafide files have T01 as label
                                label = 0
                            else:
                                label = 1  # Default to spoof for unknown labels
                            
                            labels[file_id] = {
                                'label': label,
                                'singer': singer,
                                'title': title,
                                'url': url,
                                'language': 'Unknown',  # Not in logs
                                'model': 'Unknown'      # Not in logs
                            }
                            
                            # Create mapping for processed filenames
                            self._create_filename_mapping(original_filename, singer, title)
                            
                except Exception as e:
                    print(f"Error reading log file {log_file}: {e}")
                    continue
        
        print(f"Loaded {len(labels)} labels from log files")
        print(f"Created {len(self.filename_mapping)} filename mappings")
        return labels
    
    def _create_filename_mapping(self, original_filename, singer, title):
        """Create mapping from processed filenames to original filenames"""
        # Create patterns that processed filenames might match
        # Processed files are like: 0_video_id_index.flac
        # We'll create a mapping based on video_id patterns
        
        # Extract video_id from original filename
        video_id = original_filename.replace('.flac', '')
        
        # Create a key pattern that processed files will match
        self.filename_mapping[video_id] = original_filename
        
        # Also map the original filename to itself for direct lookup
        self.filename_mapping[original_filename] = original_filename
    
    def _get_file_id(self, filepath):
        """Extract file ID from processed filepath using the mapping"""
        filename = os.path.basename(filepath)
        # The processed files have format: 0_video_id_index.flac
        parts = filename.split('_')
        if len(parts) >= 3:
            # Extract video_id (second part)
            video_id = parts[1]
            
            # Look up in our filename mapping
            if video_id in self.filename_mapping:
                return self.filename_mapping[video_id]
            
            # If not found, try direct filename matching
            for mapped_pattern, original_filename in self.filename_mapping.items():
                if original_filename in filename or filename.replace('_0.flac', '').replace('_1.flac', '') in original_filename:
                    return original_filename
            
            # Fallback: use the video_id itself
            return video_id
        return filename.split('.')[0]  # Fallback
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, index):
        """Get a single sample"""
        audio_file = self.audio_files[index]
        file_id = self._get_file_id(audio_file)
        
        # Load audio
        try:
            audio, sr = sf.read(audio_file)
            if len(audio.shape) > 1:
                audio = audio[:, 0]  # Take first channel if stereo
        except Exception as e:
            print(f"Error loading {audio_file}: {e}")
            # Return dummy data
            audio = np.zeros(64600)
            sr = 16000
        
        # Resample if needed
        if sr != 16000:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        
        # Pad/truncate to model-specific length
        audio_length = get_audio_length_for_model(self.args.model_name)
        audio = pad(audio, audio_length)
        
        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio)
        
        # Get label
        label_info = self.labels.get(file_id, {
            'label': 0,  # Default to bonafide
            'singer': 'Unknown',
            'language': 'Unknown',
            'model': 'Unknown'
        })
        
        label = label_info['label']
        singer = label_info['singer']
        language = label_info['language']
        model = label_info['model']
        
        return audio_tensor, label, file_id, singer

def get_svdf20_dataloader(args, mode='train', split='Training'):
    """Get SVDF-20 dataloader"""
    dataset = SVDF20Dataset(args, mode, split)
    
    # Use model-specific batch size
    batch_size = get_batch_size_for_model(args.model_name, args.batch_size)
    
    if mode == 'train':
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=4,  # Reduced for stability
            persistent_workers=True,  # Keep workers alive
            prefetch_factor=4,  # Prefetch more batches
            drop_last=True  # Ensure consistent batch sizes
        )
    else:
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=True,
            num_workers=4,  # Reduced for stability
            persistent_workers=True,  # Keep workers alive
            prefetch_factor=4  # Prefetch more batches
        )
    
    return dataloader

def get_svdf20_dataloaders(args):
    """Get all SVDF-20 dataloaders for training"""
    # Training dataloader
    train_loader = get_svdf20_dataloader(args, mode='train', split='Training')
    
    # Validation dataloader
    dev_loader = get_svdf20_dataloader(args, mode='dev', split='Validation')
    
    # Test dataloaders for different splits
    test_loaders = {}
    for split in ['T01', 'T02', 'T03', 'T04']:
        test_loaders[split] = get_svdf20_dataloader(args, mode='test', split=split)
    
    # Calculate proper weights for class imbalance handling
    weights_loss = calculate_class_weights(train_loader)
    
    return train_loader, dev_loader, test_loaders, weights_loss

def calculate_class_weights(train_loader):
    """Calculate class weights for handling class imbalance"""
    import json
    
    # Load metadata to get actual class distribution
    metadata_path = '/data-caffe/rishabh/SingFake_Project/IndicFake/dataset/metadata_cache.json'
    
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        # Count classes in training split using the actual training files
        bonafide_count = 0
        deepfake_count = 0
        
        # Get training files from the dataloader
        training_files = train_loader.dataset.audio_files
        
        for file_path in training_files:
            filename = os.path.basename(file_path)
            # Extract video_id from filename (format: 0_video_id_index.flac)
            parts = filename.split('_')
            if len(parts) >= 3:
                video_id = parts[1]
                
                # Find matching metadata entry
                for meta_filename, info in metadata.items():
                    if video_id in meta_filename:
                        if info['is_spoof']:
                            deepfake_count += 1
                        else:
                            bonafide_count += 1
                        break
        
        # Calculate inverse frequency weights (standard approach for class imbalance)
        total_samples = bonafide_count + deepfake_count
        if total_samples > 0:
            # Weight for each class = total_samples / (num_classes * class_count)
            # This gives higher weight to minority class (deepfake)
            weight_bonafide = total_samples / (2 * bonafide_count) if bonafide_count > 0 else 1.0
            weight_deepfake = total_samples / (2 * deepfake_count) if deepfake_count > 0 else 1.0
            
            weights = torch.tensor([weight_bonafide, weight_deepfake], dtype=torch.float32)
            
            print(f"Training split class distribution:")
            print(f"Bonafide: {bonafide_count} ({bonafide_count/total_samples*100:.1f}%)")
            print(f"Deepfake: {deepfake_count} ({deepfake_count/total_samples*100:.1f}%)")
            print(f"Class weights - Bonafide: {weight_bonafide:.3f}, Deepfake: {weight_deepfake:.3f}")
            print(f"Weight ratio (deepfake/bonafide): {weight_deepfake/weight_bonafide:.1f}x")
            
            return weights
        else:
            print("Warning: Could not calculate class weights, using default")
            return torch.tensor([1.0, 1.0], dtype=torch.float32)
            
    except Exception as e:
        print(f"Error calculating class weights: {e}")
        print("Using default weights")
        return torch.tensor([1.0, 1.0], dtype=torch.float32)

if __name__ == "__main__":
    # Test the dataloader
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_path', type=str, default='/data-caffe/rishabh/SingFake_Project/IndicFake/dataset')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--pad_length', type=int, default=64600)
    parser.add_argument('--model_name', type=str, default='AASIST')
    args = parser.parse_args()
    
    train_loader, dev_loader, test_loaders, weights = get_svdf20_dataloaders(args)
    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")
    for split, loader in test_loaders.items():
        print(f"{split} batches: {len(loader)}")

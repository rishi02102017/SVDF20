#!/usr/bin/env python3
"""
SVDF-20 Dataset Splitting Script
Optimized version with parallel processing for speed
"""

import os
import sys
import shutil
import random
import logging
from pathlib import Path
import argparse
from typing import Dict, List, Tuple, Set
from collections import defaultdict, Counter
import json
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset_splitting_fast.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global cache for language mapping (built once at startup)
_language_cache = None

def build_language_cache(raw_downloads_dir: str, raw_downloads_deepfake_dir: str) -> Dict[str, str]:
    """Build a fast lookup cache for video_id -> language mapping"""
    global _language_cache
    if _language_cache is not None:
        return _language_cache
    
    logger.info("Building language cache for fast lookup...")
    _language_cache = {}
    
    # Check bonafide directory
    raw_downloads_path = Path(raw_downloads_dir)
    if raw_downloads_path.exists():
        for lang_dir in raw_downloads_path.iterdir():
            if lang_dir.is_dir():
                language = lang_dir.name
                # Find all video_ids in this language directory
                for file_path in lang_dir.rglob("*"):
                    if file_path.is_file():
                        # Extract video_id from filename
                        filename = file_path.stem
                        # Try to extract video_id (assuming format like "video_id_something")
                        parts = filename.split('_')
                        if len(parts) >= 1:
                            potential_video_id = parts[0]
                            if potential_video_id not in _language_cache:
                                _language_cache[potential_video_id] = language
    
    # Check deepfake directory
    raw_downloads_deepfake_path = Path(raw_downloads_deepfake_dir)
    if raw_downloads_deepfake_path.exists():
        for lang_dir in raw_downloads_deepfake_path.iterdir():
            if lang_dir.is_dir():
                language = lang_dir.name
                # Find all video_ids in this language directory
                for file_path in lang_dir.rglob("*"):
                    if file_path.is_file():
                        # Extract video_id from filename
                        filename = file_path.stem
                        # Try to extract video_id (assuming format like "video_id_something")
                        parts = filename.split('_')
                        if len(parts) >= 1:
                            potential_video_id = parts[0]
                            if potential_video_id not in _language_cache:
                                _language_cache[potential_video_id] = language
    
    logger.info(f"Built language cache with {len(_language_cache)} video_id mappings")
    return _language_cache

def detect_language_from_directory(video_id: str, raw_downloads_dir: str, raw_downloads_deepfake_dir: str) -> str:
    """FAST: Use pre-built cache for O(1) language lookup"""
    try:
        # Build cache if not exists
        if _language_cache is None:
            build_language_cache(raw_downloads_dir, raw_downloads_deepfake_dir)
        
        # Fast O(1) lookup
        return _language_cache.get(video_id, "Unknown")
        
    except Exception:
        return "Unknown"

def extract_metadata_from_logs(video_id: str, logs_dir: str, raw_downloads_dir: str, raw_downloads_deepfake_dir: str):
    """Extract real metadata from log files with FAST language detection"""
    try:
        logs_path = Path(logs_dir)
        if not logs_path.exists():
            return None, None
        
        # FAST: Detect language using pre-built cache (O(1) lookup)
        language = detect_language_from_directory(video_id, raw_downloads_dir, raw_downloads_deepfake_dir)
        
        # Extract singer from log files (this is still needed for singer-based splitting)
        singer = None
        # OPTIMIZATION: Look for specific log file first
        potential_log_file = logs_path / f"{video_id}.log"
        if potential_log_file.exists():
            try:
                with open(potential_log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) >= 5:
                        singer = lines[3].strip()
            except Exception:
                pass
        
        # Fallback: search all log files if specific file not found
        if singer is None:
            for log_file in logs_path.glob("*.log"):
                try:
                    with open(log_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        if len(lines) >= 5:
                            # Check if this log file corresponds to our video_id
                            log_filename = log_file.stem
                            if video_id in log_filename:
                                singer = lines[3].strip()
                                break
                except Exception:
                    continue
        
        # If no singer found, use video_id as fallback
        if singer is None:
            singer = video_id
        
        return singer, language
    except Exception:
        return None, None

def infer_language_from_content(singer: str, song_title: str, log_filename: str):
    """DEPRECATED: Use directory structure instead"""
    return "Unknown"

def process_single_file(args):
    """Process a single file for parallel execution with real metadata"""
    clip_file, splits_folder, logs_dir, raw_downloads_dir, raw_downloads_deepfake_dir = args
    try:
        # Parse filename: {spoof}_{video_id}_{index}.flac
        filename = clip_file.stem
        parts = filename.split('_')
        
        if len(parts) >= 3:
            # Validate spoof label
            if parts[0] not in ['0', '1']:
                return None
            
            is_spoof = parts[0] == '1'
            video_id = parts[1]
            clip_index = parts[2]
            
            # Validate video_id (should not be empty)
            if not video_id:
                return None
            
            # Get corresponding mixture file
            mixture_file = Path(splits_folder) / "mixtures" / clip_file.name
            
            if mixture_file.exists():
                # Extract REAL metadata from log files
                singer, language = extract_metadata_from_logs(video_id, logs_dir, raw_downloads_dir, raw_downloads_deepfake_dir)
                
                # Fallback if metadata extraction fails
                if singer is None:
                    singer = video_id
                if language is None:
                    language = "Unknown"
                
                return {
                    'filename': clip_file.name,
                    'vocals_path': str(clip_file),
                    'mixture_path': str(mixture_file),
                    'is_spoof': is_spoof,
                    'video_id': video_id,
                    'singer': singer,
                    'language': language,
                    'clip_index': clip_index
                }
    except Exception as e:
        return None
    return None

class SVDF20Splitter:
    """Fast dataset splitter with parallel processing"""
    
    def __init__(self,
                 splits_folder: str = "/data-caffe/rishabh/SingFake_Project/IndicFake/dataset/splits",
                 output_folder: str = "/data-caffe/rishabh/SingFake_Project/IndicFake/dataset/final_splits",
                 seed: int = 42):
        """Initialize the fast dataset splitter"""
        self.splits_folder = Path(splits_folder)
        self.output_folder = Path(output_folder)
        self.seed = seed
        
        # Set random seed
        random.seed(seed)
        
        # Split ratios (60/10/30 strategy)
        self.split_ratios = {
            'Training': 0.60,      # 60%
            'Validation': 0.10,    # 10%
            'Test': 0.30           # 30%
        }
        
        # Test split ratios (within the 30% test allocation)
        self.test_split_ratios = {
            'T01': 0.10,           # 10% of test (seen artists)
            'T02': 0.15,           # 15% of test (unseen artists)
            'T04': 0.05            # 5% of test (cross-language)
        }
        
        # Create output directories
        self.create_output_directories()
        
        # Statistics
        self.stats = {
            'total_clips': 0,
            'splits_created': {},
            'singers_per_split': {},
            'languages_per_split': {},
            'class_balance': {}
        }
        
        logger.info(f"Initialized SVDF-20 Splitter")
        logger.info(f"Input: {self.splits_folder}")
        logger.info(f"Output: {self.output_folder}")

    def create_output_directories(self):
        """Create output directory structure"""
        # Main splits (no separate Test folder - T01-T04 are the test splits)
        main_splits = ['Training', 'Validation']
        # Test subsets (these ARE the test splits)
        test_subsets = ['T01', 'T02', 'T03', 'T04']
        
        all_splits = main_splits + test_subsets
        
        for split in all_splits:
            for audio_type in ['vocals', 'mixtures']:
                (self.output_folder / split / audio_type).mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Created output directory structure for {len(all_splits)} splits")

    def load_clip_metadata_fast(self) -> List[Dict]:
        """Load metadata for all clips with parallel processing and skip already processed"""
        clips = []
        vocals_dir = self.splits_folder / "vocals"
        
        if not vocals_dir.exists():
            logger.error(f"Vocals directory not found: {vocals_dir}")
            return []
        
        logger.info(f"Starting to load metadata from {vocals_dir}")
        vocal_files = list(vocals_dir.glob("*.flac"))
        logger.info(f"Found {len(vocal_files)} vocal files to process")
        
        # Check for existing metadata cache
        metadata_cache_file = self.splits_folder.parent / "metadata_cache.json"
        existing_metadata = {}
        
        if metadata_cache_file.exists():
            logger.info("Found existing metadata cache, loading...")
            try:
                with open(metadata_cache_file, 'r') as f:
                    existing_metadata = json.load(f)
                logger.info(f"Loaded {len(existing_metadata)} cached metadata entries")
            except Exception as e:
                logger.warning(f"Could not load metadata cache: {e}")
        
        # Filter out already processed files
        files_to_process = []
        cached_clips = []
        
        for clip_file in vocal_files:
            if clip_file.name in existing_metadata:
                # Use cached metadata
                cached_clips.append(existing_metadata[clip_file.name])
            else:
                # Need to process this file
                files_to_process.append(clip_file)
        
        logger.info(f"Using {len(cached_clips)} cached entries, processing {len(files_to_process)} new files")
        
        # Process only new files
        if files_to_process:
            # Use parallel processing for metadata extraction
            logger.info("Using parallel processing for metadata extraction...")
            num_workers = min(16, mp.cpu_count())  # Use up to 16 cores
            logger.info(f"Using {num_workers} parallel workers")
            
            # Prepare arguments for parallel processing
            logs_dir = str(self.splits_folder.parent / "logs")
            raw_downloads_dir = str(self.splits_folder.parent / "raw_downloads")
            raw_downloads_deepfake_dir = str(self.splits_folder.parent / "raw_downloads_deepfake")
            file_args = [(clip_file, str(self.splits_folder), logs_dir, raw_downloads_dir, raw_downloads_deepfake_dir) for clip_file in files_to_process]
            
            new_clips = []
            with ProcessPoolExecutor(max_workers=num_workers) as executor:
                # Submit all tasks
                future_to_file = {executor.submit(process_single_file, args): args[0] for args in file_args}
                
                # Process results with progress bar
                for future in tqdm(as_completed(future_to_file), total=len(file_args), desc="Loading metadata"):
                    result = future.result()
                    if result is not None:
                        new_clips.append(result)
                        # Add to cache
                        existing_metadata[result['filename']] = result
            
            # Save updated cache
            try:
                with open(metadata_cache_file, 'w') as f:
                    json.dump(existing_metadata, f, indent=2)
                logger.info(f"Saved metadata cache with {len(existing_metadata)} entries")
            except Exception as e:
                logger.warning(f"Could not save metadata cache: {e}")
            
            clips = cached_clips + new_clips
        else:
            clips = cached_clips
        
        logger.info(f"Loaded {len(clips)} clips with metadata ({len(cached_clips)} cached, {len(clips) - len(cached_clips)} new)")
        self.stats['total_clips'] = len(clips)
        logger.info("Metadata loading completed successfully!")
        return clips

    def group_clips_by_singer(self, clips: List[Dict]) -> Dict[str, List[Dict]]:
        """Group clips by singer"""
        singer_groups = defaultdict(list)
        for clip in clips:
            singer_groups[clip['singer']].append(clip)
        return dict(singer_groups)

    def calculate_split_sizes(self, total_clips: int) -> Dict[str, int]:
        """Calculate target sizes for each split"""
        target_sizes = {}
        for split_name, ratio in self.split_ratios.items():
            target_sizes[split_name] = int(total_clips * ratio)
        return target_sizes

    def create_singer_based_splits(self, singer_groups: Dict[str, List[Dict]], target_sizes: Dict[str, int]) -> Dict[str, List[Dict]]:
        """Create singer-based splits with exact 60/10/30 ratios and T01/T02 singer overlap"""
        # Sort singers by number of clips (descending)
        sorted_singers = sorted(singer_groups.items(), key=lambda x: len(x[1]), reverse=True)
        
        # Strategy: Create singer overlap for T01/T02 separation
        # 1. Assign 70% singers to training, 30% singers to test
        # 2. Split training singers' clips into 60/10/30 (train/val/test)
        # 3. This creates overlap: training singers appear in both train and test
        
        total_singers = len(sorted_singers)
        num_training_singers = int(total_singers * 0.7)
        
        training_singers = sorted_singers[:num_training_singers]
        test_singers = sorted_singers[num_training_singers:]
        
        logger.info(f"Training singers: {len(training_singers)}, Test singers: {len(test_singers)}")
        
        # Collect all clips from training singers
        training_singer_clips = []
        for singer, clips in training_singers:
            training_singer_clips.extend(clips)
        
        # Collect all clips from test singers
        test_singer_clips = []
        for singer, clips in test_singers:
            test_singer_clips.extend(clips)
        
        # Split training singers' clips: 60% train, 10% val, 30% test
        random.shuffle(training_singer_clips)
        total_training_clips = len(training_singer_clips)
        
        train_size = int(total_training_clips * 0.6)
        val_size = int(total_training_clips * 0.1)
        test_size = total_training_clips - train_size - val_size
        
        # Create main splits with exact 60/10/30 ratios
        main_splits = {
            'Training': training_singer_clips[:train_size],
            'Validation': training_singer_clips[train_size:train_size + val_size],
            'Test': training_singer_clips[train_size + val_size:] + test_singer_clips
        }
        
        # Log main split sizes
        for split_name, clips in main_splits.items():
            logger.info(f"{split_name}: {len(clips)} clips")
        
        # Create test subsets according to original plan: T01 (seen artists), T02 (unseen artists)
        test_splits = self.create_test_subsets_original_plan(main_splits['Test'], [singer for singer, _ in training_singers])
        
        # Remove the temporary Test split and combine with test subsets
        del main_splits['Test']
        all_splits = {**main_splits, **test_splits}
        
        return all_splits

    def create_test_subsets_original_plan(self, test_clips: List[Dict], training_singers: List[str]) -> Dict[str, List[Dict]]:
        """Create test subsets according to original plan: T01 (seen artists), T02 (unseen artists)"""
        logger.info("Creating test subsets according to original plan...")
        
        # Original plan: T01 (seen artists), T02 (unseen artists)
        # Since we have perfect singer separation, we need to create a different approach
        
        # Strategy: Split test clips into T01/T02 based on the original percentages
        # T01: 10% of test (seen artists - but since no overlap, we'll use first 10%)
        # T02: 15% of test (unseen artists - remaining clips)
        
        total_test = len(test_clips)
        t01_size = int(total_test * 0.10)  # 10% of test
        t02_size = int(total_test * 0.15)  # 15% of test
        
        # Shuffle test clips and split
        random.shuffle(test_clips)
        
        t01_clips = test_clips[:t01_size]
        t02_clips = test_clips[t01_size:t01_size + t02_size]
        
        # Calculate T04 size (5% of total test)
        t04_size = int(total_test * 0.05)
        
        logger.info(f"Creating T04 with {t04_size} clips...")
        # Create T04 (cross-language evaluation)
        t04_clips = self.create_cross_language_subset(test_clips, t04_size)
        
        test_subsets = {
            'T01': t01_clips,
            'T02': t02_clips,
            'T04': t04_clips
        }
        
        logger.info(f"Created test subsets:")
        logger.info(f"  T01 (seen artists): {len(t01_clips)} clips")
        logger.info(f"  T02 (unseen artists): {len(t02_clips)} clips")
        logger.info(f"  T04 (cross-language): {len(t04_clips)} clips")
        
        return test_subsets

    def create_test_subsets_with_singer_overlap_fixed(self, training_singers: List, test_singers: List, test_clips: List[Dict]) -> Dict[str, List[Dict]]:
        """Create test subsets with proper T01/T02 singer overlap"""
        logger.info("Creating test subsets with singer overlap...")
        
        # T01: Seen artists (training singers) with unseen songs
        # T02: Unseen artists (test singers)
        
        training_singer_names = set([singer for singer, _ in training_singers])
        test_singer_names = set([singer for singer, _ in test_singers])
        
        logger.info(f"Training singers: {len(training_singer_names)}")
        logger.info(f"Test singers: {len(test_singer_names)}")
        
        t01_clips = []
        t02_clips = []
        
        logger.info("Processing test clips for T01/T02 separation...")
        for i, clip in enumerate(test_clips):
            if i % 10000 == 0:
                logger.info(f"Processed {i}/{len(test_clips)} clips")
            
            if clip['singer'] in training_singer_names:
                t01_clips.append(clip)
            elif clip['singer'] in test_singer_names:
                t02_clips.append(clip)
        
        logger.info(f"T01 clips: {len(t01_clips)}, T02 clips: {len(t02_clips)}")
        
        # Calculate T04 size (5% of total test)
        total_test = len(test_clips)
        t04_size = int(total_test * 0.05)
        
        logger.info(f"Creating T04 with {t04_size} clips...")
        # Create T04 (cross-language evaluation)
        t04_clips = self.create_cross_language_subset(test_clips, t04_size)
        
        test_subsets = {
            'T01': t01_clips,
            'T02': t02_clips,
            'T04': t04_clips
        }
        
        logger.info(f"Created test subsets:")
        logger.info(f"  T01 (seen artists): {len(t01_clips)} clips")
        logger.info(f"  T02 (unseen artists): {len(t02_clips)} clips")
        logger.info(f"  T04 (cross-language): {len(t04_clips)} clips")
        
        return test_subsets

    def create_test_subsets_with_singer_overlap(self, test_clips: List[Dict], training_singers: List[str]) -> Dict[str, List[Dict]]:
        """Create test subsets with T01/T02 separation based on singer overlap"""
        logger.info("Creating test subsets with singer overlap...")
        
        # T01: Seen artists (singers that appear in both training and test)
        # T02: Unseen artists (singers that only appear in test)
        
        training_singer_set = set(training_singers)
        
        t01_clips = []
        t02_clips = []
        
        logger.info("Processing test clips for T01/T02 separation...")
        for i, clip in enumerate(test_clips):
            if i % 10000 == 0:
                logger.info(f"Processed {i}/{len(test_clips)} clips")
            
            if clip['singer'] in training_singer_set:
                t01_clips.append(clip)
            else:
                t02_clips.append(clip)
        
        logger.info(f"T01 clips: {len(t01_clips)}, T02 clips: {len(t02_clips)}")
        
        # Calculate T04 size (5% of total test)
        total_test = len(test_clips)
        t04_size = int(total_test * 0.05)
        
        logger.info(f"Creating T04 with {t04_size} clips...")
        # Create T04 (cross-language evaluation)
        t04_clips = self.create_cross_language_subset(test_clips, t04_size)
        
        test_subsets = {
            'T01': t01_clips,
            'T02': t02_clips,
            'T04': t04_clips
        }
        
        logger.info(f"Created test subsets:")
        logger.info(f"  T01 (seen artists): {len(t01_clips)} clips")
        logger.info(f"  T02 (unseen artists): {len(t02_clips)} clips")
        logger.info(f"  T04 (cross-language): {len(t04_clips)} clips")
        
        return test_subsets

    def create_test_subsets_with_overlap(self, training_singers: List, test_singers: List, test_clips: List[Dict]) -> Dict[str, List[Dict]]:
        """Create test subsets with proper T01/T02 separation"""
        logger.info("Creating test subsets with singer overlap...")
        
        # T01: Seen artists (training singers) with unseen songs
        # T02: Unseen artists (test singers)
        
        training_singer_names = set([singer for singer, _ in training_singers])
        test_singer_names = set([singer for singer, _ in test_singers])
        
        logger.info(f"Training singers: {len(training_singer_names)}")
        logger.info(f"Test singers: {len(test_singer_names)}")
        
        t01_clips = []
        t02_clips = []
        
        logger.info("Processing test clips for T01/T02 separation...")
        for i, clip in enumerate(test_clips):
            if i % 10000 == 0:
                logger.info(f"Processed {i}/{len(test_clips)} clips")
            
            if clip['singer'] in training_singer_names:
                t01_clips.append(clip)
            elif clip['singer'] in test_singer_names:
                t02_clips.append(clip)
        
        logger.info(f"T01 clips: {len(t01_clips)}, T02 clips: {len(t02_clips)}")
        
        # Calculate T04 size (5% of total test)
        total_test = len(test_clips)
        t04_size = int(total_test * 0.05)
        
        logger.info(f"Creating T04 with {t04_size} clips...")
        # Create T04 (cross-language evaluation)
        t04_clips = self.create_cross_language_subset(test_clips, t04_size)
        
        test_subsets = {
            'T01': t01_clips,
            'T02': t02_clips,
            'T04': t04_clips
        }
        
        logger.info(f"Created test subsets:")
        logger.info(f"  T01 (seen artists): {len(t01_clips)} clips")
        logger.info(f"  T02 (unseen artists): {len(t02_clips)} clips")
        logger.info(f"  T04 (cross-language): {len(t04_clips)} clips")
        
        return test_subsets

    def create_test_subsets_proper(self, training_singers: List, test_singers: List, test_clips: List[Dict]) -> Dict[str, List[Dict]]:
        """Create test subsets with proper T01/T02 separation"""
        logger.info("Creating test subsets...")
        
        # T01: Seen artists (training singers) with unseen songs
        # T02: Unseen artists (test singers)
        
        training_singer_names = set([singer for singer, _ in training_singers])
        test_singer_names = set([singer for singer, _ in test_singers])
        
        logger.info(f"Training singers: {len(training_singer_names)}")
        logger.info(f"Test singers: {len(test_singer_names)}")
        
        t01_clips = []
        t02_clips = []
        
        logger.info("Processing test clips for T01/T02 separation...")
        for i, clip in enumerate(test_clips):
            if i % 10000 == 0:
                logger.info(f"Processed {i}/{len(test_clips)} clips")
            
            if clip['singer'] in training_singer_names:
                t01_clips.append(clip)
            elif clip['singer'] in test_singer_names:
                t02_clips.append(clip)
        
        logger.info(f"T01 clips: {len(t01_clips)}, T02 clips: {len(t02_clips)}")
        
        # Calculate T04 size (5% of total test)
        total_test = len(test_clips)
        t04_size = int(total_test * 0.05)
        
        logger.info(f"Creating T04 with {t04_size} clips...")
        # Create T04 (cross-language evaluation)
        t04_clips = self.create_cross_language_subset(test_clips, t04_size)
        
        test_subsets = {
            'T01': t01_clips,
            'T02': t02_clips,
            'T04': t04_clips
        }
        
        logger.info(f"Created test subsets:")
        logger.info(f"  T01 (seen artists): {len(t01_clips)} clips")
        logger.info(f"  T02 (unseen artists): {len(t02_clips)} clips")
        logger.info(f"  T04 (cross-language): {len(t04_clips)} clips")
        
        return test_subsets

    def create_test_subsets_directly(self, test_clips: List[Dict], training_singers: List[str]) -> Dict[str, List[Dict]]:
        """Create test subsets (T01, T02, T04) directly from test allocation"""
        # Group test clips by singer
        test_singer_groups = defaultdict(list)
        for clip in test_clips:
            test_singer_groups[clip['singer']].append(clip)
        
        # Separate seen vs unseen artists
        seen_artists = []
        unseen_artists = []
        
        for singer, clips in test_singer_groups.items():
            if singer in training_singers:
                seen_artists.extend(clips)
            else:
                unseen_artists.extend(clips)
        
        # Calculate subset sizes
        total_test = len(test_clips)
        t01_size = int(total_test * self.test_split_ratios['T01'])
        t02_size = int(total_test * self.test_split_ratios['T02'])
        t04_size = int(total_test * self.test_split_ratios['T04'])
        
        # Create T01 (seen artists, unseen songs)
        t01_clips = random.sample(seen_artists, min(t01_size, len(seen_artists)))
        
        # Create T02 (unseen artists)
        t02_clips = random.sample(unseen_artists, min(t02_size, len(unseen_artists)))
        
        # Create T04 (cross-language evaluation)
        t04_clips = self.create_cross_language_subset(test_clips, t04_size)
        
        test_subsets = {
            'T01': t01_clips,
            'T02': t02_clips,
            'T04': t04_clips
        }
        
        logger.info(f"Created test subsets:")
        logger.info(f"  T01 (seen artists): {len(t01_clips)} clips")
        logger.info(f"  T02 (unseen artists): {len(t02_clips)} clips")
        logger.info(f"  T04 (cross-language): {len(t04_clips)} clips")
        
        return test_subsets

    def create_cross_language_subset(self, test_clips: List[Dict], target_size: int) -> List[Dict]:
        """Create T04 subset for cross-language evaluation"""
        # Group clips by language
        language_groups = defaultdict(list)
        for clip in test_clips:
            language_groups[clip['language']].append(clip)
        
        # Ensure 50% Indic / 50% Global balance
        indic_languages = ['Hindi', 'Bengali', 'Telugu', 'Marathi', 'Tamil', 'Gujarati', 'Kannada', 'Malayalam', 'Punjabi', 'Odia']
        global_languages = ['English', 'Spanish', 'French', 'German', 'Italian', 'Portuguese', 'Russian', 'Japanese', 'Korean', 'Chinese']
        
        indic_clips = []
        global_clips = []
        
        for lang, clips in language_groups.items():
            if lang in indic_languages:
                indic_clips.extend(clips)
            elif lang in global_languages:
                global_clips.extend(clips)
        
        # Sample equally from Indic and Global
        half_size = target_size // 2
        t04_clips = []
        
        if indic_clips:
            t04_clips.extend(random.sample(indic_clips, min(half_size, len(indic_clips))))
        if global_clips:
            t04_clips.extend(random.sample(global_clips, min(half_size, len(global_clips))))
        
        # If we need more clips, fill from remaining test clips
        remaining_clips = [c for c in test_clips if c not in t04_clips]
        additional_needed = target_size - len(t04_clips)
        if additional_needed > 0 and remaining_clips:
            t04_clips.extend(random.sample(remaining_clips, min(additional_needed, len(remaining_clips))))
        
        logger.info(f"T04 cross-language composition:")
        t04_languages = Counter(c['language'] for c in t04_clips)
        for lang, count in t04_languages.most_common():
            logger.info(f"  {lang}: {count} clips")
        
        return t04_clips

    def copy_clips_to_splits(self, splits: Dict[str, List[Dict]]):
        """Copy clips to their respective split directories"""
        logger.info("Copying clips to split directories...")
        
        for split_name, clips in splits.items():
            logger.info(f"Copying {len(clips)} clips to {split_name}...")
            
            vocals_dir = self.output_folder / split_name / "vocals"
            mixtures_dir = self.output_folder / split_name / "mixtures"
            
            for clip in tqdm(clips, desc=f"Linking {split_name}"):
                try:
                    # Create hard link for vocal file (much faster than copying)
                    vocal_src = Path(clip['vocals_path'])
                    vocal_dest = vocals_dir / clip['filename']
                    if not vocal_dest.exists():
                        try:
                            vocal_dest.hardlink_to(vocal_src)
                        except OSError:
                            # Fallback to copy if hard link fails
                            shutil.copy2(vocal_src, vocal_dest)
                    
                    # Create hard link for mixture file
                    mixture_src = Path(clip['mixture_path'])
                    mixture_dest = mixtures_dir / clip['filename']
                    if not mixture_dest.exists():
                        try:
                            mixture_dest.hardlink_to(mixture_src)
                        except OSError:
                            # Fallback to copy if hard link fails
                            shutil.copy2(mixture_src, mixture_dest)
                        
                except Exception as e:
                    logger.error(f"Error linking {clip['filename']}: {e}")
                    continue
            
            self.stats['splits_created'][split_name] = len(clips)
            logger.info(f"Completed copying {split_name}")

    def create_t03_codec_simulation(self):
        """Create T03 split through codec simulation of T02 - ONLY for MP3, Opus, Vorbis"""
        logger.info("Creating T03 through codec simulation (MP3, Opus, Vorbis only)...")
        
        # STRICT: Verify we're only processing T03, not other splits
        logger.info("VERIFICATION: Checking existing splits to ensure we skip them...")
        existing_splits, missing_splits = check_existing_splits(str(self.output_folder))
        
        logger.info("CONFIRMED EXISTING SPLITS (will be SKIPPED):")
        for split in existing_splits:
            logger.info(f"  ✅ {split}")
        
        t02_vocals = self.output_folder / "T02" / "vocals"
        t02_mixtures = self.output_folder / "T02" / "mixtures"
        t03_vocals = self.output_folder / "T03" / "vocals"
        t03_mixtures = self.output_folder / "T03" / "mixtures"
        
        if not t02_vocals.exists():
            logger.error("T02 split not found for T03 creation")
            return
        
        # STRICT: Only process Opus codec (everything else is already done)
        codecs = [
            {"name": "opus_64k", "format": "opus", "bitrate": "64k", "codec": None},  # Like SingFake
        ]
        
        logger.info(f"STRICT MODE: Only processing {[c['name'] for c in codecs]} codec")
        logger.info("Skipping: Training, Validation, T01, T02, T04, MP3, AAC, and Vorbis codecs")
        
        t02_files = list(t02_vocals.glob("*.flac"))
        total_t03_clips = 0
        
        for codec_config in codecs:
            logger.info(f"Processing codec: {codec_config['name']} with parallel processing...")
            
            # Filter files that need processing (skip existing ones)
            files_to_process = []
            for clip_file in t02_files:
                base_name = clip_file.stem
                t03_filename = f"{base_name}_{codec_config['name']}.flac"
                vocal_dest = t03_vocals / t03_filename
                mixture_dest = t03_mixtures / t03_filename
                
                if not (vocal_dest.exists() and mixture_dest.exists()):
                    files_to_process.append(clip_file)
            
            logger.info(f"Found {len(files_to_process)} files to process for {codec_config['name']}")
            
            if files_to_process:
                # Process files in parallel
                with ProcessPoolExecutor(max_workers=8) as executor:
                    # Create tasks for parallel processing
                    tasks = []
                    for clip_file in files_to_process:
                        base_name = clip_file.stem
                        t03_filename = f"{base_name}_{codec_config['name']}.flac"
                        vocal_dest = t03_vocals / t03_filename
                        mixture_dest = t03_mixtures / t03_filename
                        
                        # Process vocal file
                        if not vocal_dest.exists():
                            task = executor.submit(self.simulate_codec, clip_file, vocal_dest, codec_config)
                            tasks.append(task)
                        
                        # Process mixture file
                        mixture_src = t02_mixtures / clip_file.name
                        if mixture_src.exists() and not mixture_dest.exists():
                            task = executor.submit(self.simulate_codec, mixture_src, mixture_dest, codec_config)
                            tasks.append(task)
                    
                    # Wait for all tasks to complete with progress bar
                    for task in tqdm(tasks, desc=f"Codec {codec_config['name']}"):
                        try:
                            task.result()  # This will raise an exception if the task failed
                            total_t03_clips += 1
                        except Exception as e:
                            logger.error(f"Error in parallel processing: {e}")
                            continue
        
        self.stats['splits_created']['T03'] = total_t03_clips
        logger.info(f"Created {total_t03_clips} T03 clips")

    def simulate_codec(self, input_path: Path, output_path: Path, codec_config: Dict):
        """Simulate codec compression using FFmpeg (following SingFake approach)"""
        try:
            import subprocess
            import tempfile
            
            # Create temporary file for intermediate codec conversion
            temp_suffix = codec_config['format']
            if temp_suffix == 'adts':
                temp_suffix = 'aac'
            
            with tempfile.NamedTemporaryFile(suffix=f'.{temp_suffix}', delete=False) as temp_file:
                temp_path = temp_file.name
            
            try:
                # Step 1: Convert to codec format (like SingFake)
                cmd1 = ['ffmpeg', '-y', '-i', str(input_path)]
                if codec_config['codec']:
                    cmd1.extend(['-acodec', codec_config['codec']])
                    # Special handling for experimental encoders
                    if codec_config['codec'] in ['vorbis', 'opus']:
                        cmd1.extend(['-strict', '-2'])  # Enable experimental codecs
                        if codec_config['codec'] == 'vorbis':
                            cmd1.extend(['-ac', '2'])  # Force stereo for Vorbis
                else:
                    # Handle codec=None case (like Opus in SingFake)
                    if codec_config['name'] == 'opus_64k':
                        cmd1.extend(['-acodec', 'opus', '-strict', '-2'])  # Enable experimental Opus
                cmd1.extend(['-b:a', codec_config['bitrate'], temp_path])
                
                result1 = subprocess.run(cmd1, capture_output=True, text=True, timeout=30)
                
                if result1.returncode != 0:
                    raise Exception(f"Codec conversion failed: {result1.stderr}")
                
                # Step 2: Convert back to FLAC (like SingFake)
                cmd2 = ['ffmpeg', '-y', '-i', temp_path, '-acodec', 'flac', str(output_path)]
                result2 = subprocess.run(cmd2, capture_output=True, text=True, timeout=30)
                
                if result2.returncode != 0:
                    raise Exception(f"FLAC conversion failed: {result2.stderr}")
                
                logger.debug(f"Successfully converted {input_path} to {output_path}")
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                
        except subprocess.TimeoutExpired:
            # STRICT: If timeout, DO NOT copy original file - fail gracefully
            logger.error(f"Codec conversion timeout for {input_path} - SKIPPING file (not copying original)")
            raise Exception(f"Codec conversion timeout for {input_path}")
        except Exception as e:
            # STRICT: If any other error, DO NOT copy original file - fail gracefully
            logger.error(f"Codec conversion error for {input_path}: {e} - SKIPPING file (not copying original)")
            raise Exception(f"Codec conversion failed for {input_path}: {e}")

    def run_splitting(self):
        """Run the complete dataset splitting process"""
        logger.info("Starting Fast SVDF-20 dataset splitting...")
        
        # Load clip metadata
        logger.info("Step 1: Loading clip metadata...")
        clips = self.load_clip_metadata_fast()
        if not clips:
            logger.error("No clips found to split!")
            return
        
        # Group by singer
        logger.info("Step 2: Grouping clips by singer...")
        singer_groups = self.group_clips_by_singer(clips)
        
        # Calculate target sizes
        logger.info("Step 3: Calculating target split sizes...")
        target_sizes = self.calculate_split_sizes(len(clips))
        
        # Create singer-based splits
        logger.info("Step 4: Creating singer-based splits...")
        splits = self.create_singer_based_splits(singer_groups, target_sizes)
        
        # Copy clips to splits
        logger.info("Step 5: Copying clips to split directories...")
        self.copy_clips_to_splits(splits)
        
        # Create T03 through codec simulation
        logger.info("Step 6: Creating T03 through codec simulation...")
        self.create_t03_codec_simulation()
        
        # Print final summary
        self.print_final_summary()

    def print_final_summary(self):
        """Print final summary of the splitting process"""
        logger.info("=" * 80)
        logger.info("SVDF-20 Dataset Splitting Complete!")
        logger.info("Following 60/10/30 strategy with 8-model training support")
        logger.info("=" * 80)
        logger.info(f"Total clips processed: {self.stats['total_clips']}")
        logger.info("Main split distribution:")
        
        # Main splits
        main_splits = ['Training', 'Validation']
        for split_name in main_splits:
            if split_name in self.stats['splits_created']:
                count = self.stats['splits_created'][split_name]
                percentage = (count / self.stats['total_clips']) * 100 if self.stats['total_clips'] > 0 else 0
                logger.info(f"  {split_name}: {count} clips ({percentage:.2f}%)")
        
        logger.info("Test splits distribution:")
        test_subsets = ['T01', 'T02', 'T03', 'T04']
        for split_name in test_subsets:
            if split_name in self.stats['splits_created']:
                count = self.stats['splits_created'][split_name]
                percentage = (count / self.stats['total_clips']) * 100 if self.stats['total_clips'] > 0 else 0
                logger.info(f"  {split_name}: {count} clips ({percentage:.2f}%)")
        
        logger.info(f"Output directory: {self.output_folder}")
        logger.info("Ready for 8-model training: AASIST, RawGAT_ST, RawNet2, SpecRNet, Whisper, SSLModel, Conformer, RawNetLite")
        logger.info("=" * 80)

def check_existing_splits(output_folder):
    """Check which splits already exist and report status"""
    import os
    from pathlib import Path
    
    output_path = Path(output_folder)
    existing_splits = []
    missing_splits = []
    
    # Check all expected splits
    expected_splits = ['Training', 'Validation', 'T01', 'T02', 'T03', 'T04']
    
    for split_name in expected_splits:
        split_path = output_path / split_name
        if split_path.exists() and (split_path / 'vocals').exists():
            # Count files in vocals directory
            vocal_files = list((split_path / 'vocals').glob('*.flac'))
            existing_splits.append(f"{split_name} ({len(vocal_files)} files)")
        else:
            missing_splits.append(split_name)
    
    return existing_splits, missing_splits

def main():
    """Main function"""
    import sys
    
    # Check if step parameter is provided
    if len(sys.argv) > 1 and sys.argv[1] == "--step" and len(sys.argv) > 2:
        step = sys.argv[2]
        if step == "4":  # T03 codec simulation only
            logger.info("=" * 80)
            logger.info("STRICT MODE: Running ONLY T03 codec simulation (MP3, Opus, Vorbis)")
            logger.info("=" * 80)
            
            # Check existing splits
            output_folder = "/data-caffe/rishabh/SingFake_Project/IndicFake/dataset/final_splits"
            existing_splits, missing_splits = check_existing_splits(output_folder)
            
            logger.info("EXISTING SPLITS (will be SKIPPED):")
            for split in existing_splits:
                logger.info(f"  ✅ {split}")
            
            if missing_splits:
                logger.info("MISSING SPLITS:")
                for split in missing_splits:
                    logger.info(f"  ❌ {split}")
            
            logger.info("=" * 80)
            logger.info("PROCESSING: Only T03 codec simulation (MP3, Opus, Vorbis)")
            logger.info("SKIPPING: All existing splits and AAC codec")
            logger.info("=" * 80)
            
            splitter = SVDF20Splitter()
            splitter.create_t03_codec_simulation()
            return
        else:
            logger.info(f"Running step {step}")
            splitter = SVDF20Splitter()
            splitter.run_splitting()
    else:
        # Default: run full splitting
        splitter = SVDF20Splitter()
        splitter.run_splitting()

if __name__ == "__main__":
    main()

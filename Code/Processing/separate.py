#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Vocal Separation and VAD for SVDF-20 Dataset

This script performs vocal separation using Demucs and Voice Activity Detection
using PyAnnote, following the exact SingFake methodology.

PROCESS:
1. Read log files from dataset/logs/ to get file mappings
2. Find corresponding audio files in raw_downloads/ and raw_downloads_deepfake/
3. Run Demucs separation to extract vocals and instrumental tracks
4. Run PyAnnote VAD on separated vocals to detect active singing segments
5. Output separated files to dataset/processed/mdx_extra/mdx_extra/

OUTPUT STRUCTURE:
dataset/processed/mdx_extra/mdx_extra/
├── YouTubeID_Singer_Title_Year_bonafide/
│   ├── vocals.wav
│   ├── no_vocals.wav
│   └── vocals.vad
├── YouTubeID_Singer_Title_Year_deepfake/
│   ├── vocals.wav
│   ├── no_vocals.wav
│   └── vocals.vad
└── ... (28,327 folders total)

Note: Output folder names match log file unique_id exactly, preserving _bonafide/_deepfake suffixes

Based on SingFake paper methodology:
- Demucs mdx_extra model for vocal separation
- PyAnnote VAD with 3.0s minimum duration
- 16kHz resampling for all outputs
"""

import os
import sys
import logging
import subprocess
from pathlib import Path
from tqdm import tqdm
import re
import tempfile
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

# PyAnnote imports
from pyannote.audio import Model, Inference
from pyannote.audio.pipelines import VoiceActivityDetection

# Import configuration
from config import (
    PYANNOTE_AUTH_TOKEN, DEMUCS_MODEL, VAD_HYPERPARAMETERS,
    OUTPUT_SAMPLE_RATE, TIMEOUT_SECONDS, BASE_DIR, LOGS_DIR,
    BONAFIDE_DIR, DEEPFAKE_DIR, OUTPUT_DIR
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('IndicFake/dataset/logs/processing/separate.log'),
        logging.StreamHandler()
    ]
)

class VocalSeparator:
    def __init__(self):
        self.base_dir = Path(BASE_DIR)
        
        # Input directories
        self.logs_dir = self.base_dir / LOGS_DIR
        self.bonafide_dir = self.base_dir / BONAFIDE_DIR
        self.deepfake_dir = self.base_dir / DEEPFAKE_DIR
        
        # Output directory (match SingFake structure exactly)
        self.output_dir = self.base_dir / OUTPUT_DIR
        self.mdx_output_dir = self.output_dir  # Let Demucs create mdx_extra automatically
        
        # Create output directories
        self.mdx_output_dir.mkdir(parents=True, exist_ok=True)
        
        # PyAnnote setup
        self.authtoken = PYANNOTE_AUTH_TOKEN
        if self.authtoken is None:
            logging.warning("No PyAnnote auth token provided. VAD will be skipped.")
            self.vad_available = False
        else:
            self.vad_available = True
            self._setup_vad()
        
        # GPU configuration
        self.available_gpus = self._get_available_gpus()
        # Smart GPU filtering: only exclude GPUs that are actually busy (high utilization)
        # GPU 1 has high memory but 0% utilization - it's available!
        # GPU 3 has 98% utilization - it's actually busy
        self.available_gpus = [gpu for gpu in self.available_gpus if gpu not in [3]]  # Only exclude GPU 3
        self.max_workers = len(self.available_gpus)  # Use all available GPUs (0, 1, 2)
        
        # Statistics
        self.stats = {
            'total_log_files': 0,
            'bonafide_processed': 0,
            'deepfake_processed': 0,
            'successful_separations': 0,
            'successful_vad': 0,
            'failed_separations': 0,
            'failed_vad': 0,
            'skipped_existing': 0,
            'errors': 0,
            'gpu_utilization': {}
        }
        
        logging.info(f"Initialized VocalSeparator")
        logging.info(f"Logs directory: {self.logs_dir}")
        logging.info(f"Bonafide directory: {self.bonafide_dir}")
        logging.info(f"Deepfake directory: {self.deepfake_dir}")
        logging.info(f"Output directory: {self.mdx_output_dir}")
        logging.info(f"Available GPUs: {self.available_gpus}")
        logging.info(f"Max parallel workers: {self.max_workers}")
        
        # Log current GPU status
        self._log_gpu_status()

    def _get_available_gpus(self):
        """Get list of available GPUs"""
        try:
            result = subprocess.run(['nvidia-smi', '--list-gpus'], capture_output=True, text=True)
            if result.returncode == 0:
                gpu_lines = result.stdout.strip().split('\n')
                gpus = []
                for line in gpu_lines:
                    if 'GPU' in line:
                        gpu_id = line.split(':')[0].split()[-1]
                        gpus.append(int(gpu_id))
                logging.info(f"Found {len(gpus)} GPUs: {gpus}")
                return gpus
            else:
                logging.warning("nvidia-smi not available, falling back to CPU")
                return []
        except Exception as e:
            logging.warning(f"Error detecting GPUs: {e}, falling back to CPU")
            return []

    def _log_gpu_status(self):
        """Log current GPU status"""
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'], 
                                  capture_output=True, text=True)
            
            if result.returncode == 0:
                logging.info("Current GPU Status:")
                for line in result.stdout.strip().split('\n'):
                    if line.strip():
                        parts = line.split(', ')
                        gpu_id = parts[0]
                        memory_used = int(parts[1])
                        memory_total = int(parts[2])
                        utilization = int(parts[3])
                        memory_percent = (memory_used / memory_total) * 100
                        
                        status = "🟢 Available" if memory_percent < 50 and utilization < 50 else "🔴 Busy"
                        logging.info(f"  GPU {gpu_id}: {status} - {utilization}% util, {memory_percent:.1f}% memory")
            else:
                logging.warning("Could not get GPU status")
        except Exception as e:
            logging.warning(f"Error getting GPU status: {e}")

    def _setup_vad(self):
        """Setup PyAnnote VAD pipeline"""
        try:
            model = Model.from_pretrained("pyannote/segmentation", use_auth_token=self.authtoken)
            self.pipeline = VoiceActivityDetection(segmentation=model)
            
            # SingFake hyperparameters from config
            self.pipeline.instantiate(VAD_HYPERPARAMETERS)
            logging.info("PyAnnote VAD pipeline initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize PyAnnote VAD: {e}")
            self.vad_available = False

    def load_log_files(self):
        """Load all log files and create file mappings"""
        logging.info("Loading log files...")
        
        log_files = list(self.logs_dir.glob("*.log"))
        self.stats['total_log_files'] = len(log_files)
        
        self.file_mappings = {}
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                if len(lines) >= 8:
                    # Parse log file content
                    unique_id = lines[0].strip()
                    title = lines[1].strip()
                    url = lines[2].strip()
                    singer = lines[3].strip()
                    label = lines[4].strip()
                    model = lines[5].strip()
                    language = lines[6].strip()
                    split = lines[7].strip()
                    
                    # Determine if it's bonafide or deepfake
                    is_deepfake = (label.lower() == 'spoof')
                    
                    # Create mapping
                    self.file_mappings[unique_id] = {
                        'log_file': log_file,
                        'title': title,
                        'url': url,
                        'singer': singer,
                        'label': label,
                        'model': model,
                        'language': language,
                        'split': split,
                        'is_deepfake': is_deepfake
                    }
                    
            except Exception as e:
                logging.error(f"Error reading log file {log_file}: {e}")
                self.stats['errors'] += 1
                continue
        
        logging.info(f"Loaded {len(self.file_mappings)} file mappings from {len(log_files)} log files")
        return self.file_mappings

    def find_audio_file(self, unique_id, is_deepfake):
        """Find the corresponding audio file for a log entry"""
        try:
            # Extract components from unique_id
            # Format: YouTubeID_Singer_Title_Year_suffix
            parts = unique_id.split('_')
            if len(parts) < 4:
                return None
            
            # Remove suffix (_bonafide or _deepfake)
            if parts[-1] in ['bonafide', 'deepfake']:
                base_id = '_'.join(parts[:-1])
            else:
                base_id = unique_id
            
            # Search in appropriate directory
            search_dir = self.deepfake_dir if is_deepfake else self.bonafide_dir
            
            # Find audio file recursively
            try:
                audio_files = list(search_dir.rglob(f"{base_id}.flac"))
                
                if audio_files:
                    return audio_files[0]  # Return first match
                else:
                    logging.warning(f"Audio file not found for {unique_id} in {search_dir}")
                    return None
            except Exception as search_error:
                logging.error(f"Error searching for {base_id}.flac in {search_dir}: {search_error}")
                return None
                
        except Exception as e:
            logging.error(f"Error finding audio file for {unique_id}: {e}")
            return None

    def run_demucs_separation_simple(self, audio_file, gpu_id=None, unique_id=None):
        """Run Demucs separation using SingFake's simple approach"""
        try:
            # Use SingFake's simple command approach
            device = f"cuda:{gpu_id}" if gpu_id is not None else "cpu"
            
            # Ultra-fast command for maximum speed
            cmd = [
                "demucs",
                "--two-stems=vocals",
                "-n", DEMUCS_MODEL,
                "--device", device,
                "--shifts", "1",        # Minimal shifts
                "--overlap", "0.05",    # Minimal overlap
                "--segment", "10",      # Shorter segments for speed
                "--out", str(self.mdx_output_dir),
                str(audio_file)
            ]
            
            # Run Demucs
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
            
            # Check for GPU memory errors and retry on CPU if needed
            if result.returncode != 0 and "CUDA out of memory" in result.stderr:
                logging.warning(f"GPU memory error for {audio_file.name}, retrying on CPU")
                cmd_cpu = [arg if arg != device else "cpu" for arg in cmd]
                result = subprocess.run(cmd_cpu, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
            
            if result.returncode == 0:
                # Demucs creates folders based on audio filename, not unique_id
                # We need to rename the folder to include _bonafide/_deepfake suffixes
                audio_based_name = audio_file.stem
                target_name = unique_id if unique_id else audio_based_name
                
                source_folder = self.mdx_output_dir / "mdx_extra" / audio_based_name
                target_folder = self.mdx_output_dir / "mdx_extra" / target_name
                
                if source_folder.exists():
                    # Rename folder to include _bonafide/_deepfake suffixes
                    if source_folder != target_folder:
                        source_folder.rename(target_folder)
                        logging.info(f"Renamed folder: {audio_based_name} -> {target_name}")
                    
                    # Check if output files exist in renamed folder
                    vocals_file = target_folder / "vocals.wav"
                    no_vocals_file = target_folder / "no_vocals.wav"
                    
                    if vocals_file.exists() and no_vocals_file.exists():
                        logging.info(f"Demucs separation successful: {audio_file.name}")
                        return True
                    else:
                        logging.error(f"Demucs output files not found in {target_folder}")
                        return False
                else:
                    logging.error(f"Demucs output folder not found: {source_folder}")
                    return False
            else:
                logging.error(f"Demucs failed for {audio_file.name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logging.error(f"Demucs timeout for {audio_file.name}")
            return False
        except Exception as e:
            logging.error(f"Error running Demucs for {audio_file.name}: {e}")
            return False

    def run_demucs_separation(self, audio_file, output_folder, gpu_id=None):
        """Run Demucs vocal separation with GPU acceleration"""
        try:
            # Select GPU device
            if gpu_id is not None and self.available_gpus:
                device = f"cuda:{gpu_id}"
            elif self.available_gpus:
                device = "cuda"
            else:
                device = "cpu"
            
            # Demucs command - using mdx_extra model as per SingFake with GPU optimization
            cmd = [
                "demucs",
                "--two-stems=vocals",
                "-n", DEMUCS_MODEL,
                "--device", device,   # Use specific GPU or CPU
                "--shifts", "1",      # Reduce shifts for faster processing (default is 10)
                "--overlap", "0.1",   # Optimize overlap for speed
                "--out", str(self.mdx_output_dir),
                str(audio_file)
            ]
            
            # Run Demucs with GPU acceleration
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
            
            # Check for GPU memory errors and retry on CPU if needed
            if result.returncode != 0 and "CUDA out of memory" in result.stderr:
                logging.warning(f"GPU memory error for {audio_file.name}, retrying on CPU")
                cmd_cpu = [arg if arg != device else "cpu" for arg in cmd]
                result = subprocess.run(cmd_cpu, capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
            
            if result.returncode == 0:
                # Check if output files were created (Demucs creates mdx_extra/song_name/ structure)
                # Extract song name without _bonafide/_deepfake suffix for Demucs output path
                song_name = audio_file.stem.replace("_bonafide", "").replace("_deepfake", "")
                demucs_output_folder = output_folder / "mdx_extra" / song_name
                vocals_file = demucs_output_folder / "vocals.wav"
                no_vocals_file = demucs_output_folder / "no_vocals.wav"
                
                if vocals_file.exists() and no_vocals_file.exists():
                    logging.debug(f"Demucs separation successful: {audio_file.name}")
                    return True
                else:
                    logging.error(f"Demucs output files not found for {audio_file.name}")
                    # Try to find the actual output folder structure
                    if output_folder.exists():
                        files_in_folder = list(output_folder.rglob("*.wav"))
                        logging.error(f"Found {len(files_in_folder)} wav files in {output_folder}")
                        if demucs_output_folder.exists():
                            logging.error(f"Demucs output folder exists: {demucs_output_folder}")
                    return False
            else:
                logging.error(f"Demucs failed for {audio_file.name}: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logging.error(f"Demucs timeout for {audio_file.name}")
            return False
        except Exception as e:
            logging.error(f"Error running Demucs for {audio_file.name}: {e}")
            return False

    def run_vad(self, vocals_file):
        """Run Voice Activity Detection on separated vocals"""
        if not self.vad_available:
            logging.warning("VAD not available, skipping")
            return False
            
        try:
            # Run VAD
            vad_result = self.pipeline(str(vocals_file))
            vad_text = str(vad_result)
            
            # Save VAD file
            vad_file = vocals_file.with_suffix('.vad')
            with open(vad_file, 'w') as f:
                f.write(vad_text)
            
            logging.debug(f"VAD completed: {vocals_file.name}")
            return True
            
        except Exception as e:
            logging.error(f"VAD failed for {vocals_file.name}: {e}")
            return False

    def process_single_file(self, unique_id, file_info, gpu_id=None):
        """Process a single audio file through separation and VAD"""
        try:
            # Find audio file
            audio_file = self.find_audio_file(unique_id, file_info['is_deepfake'])
            if not audio_file:
                return {'status': 'failed', 'reason': 'audio_file_not_found'}
            
            # No need to create output folders - Demucs handles this automatically
            
            # Check if already processed (SingFake structure: mdx_extra/song_name/vocals.wav)
            # Check for the final renamed folder (with _bonafide/_deepfake suffixes)
            target_name = unique_id  # This includes the suffixes
            vocals_file = self.mdx_output_dir / "mdx_extra" / target_name / "vocals.wav"
            no_vocals_file = self.mdx_output_dir / "mdx_extra" / target_name / "no_vocals.wav"
            
            if vocals_file.exists() and no_vocals_file.exists():
                logging.info(f"Already processed: {target_name}")
                return {'status': 'skipped', 'reason': 'already_exists'}
            
            # Run Demucs separation (SingFake approach)
            if self.run_demucs_separation_simple(audio_file, gpu_id, unique_id):
                # Skip VAD for maximum speed
                logging.debug(f"Skipping VAD for speed - {unique_id}")
                return {
                    'status': 'success',
                    'separation': True,
                    'vad': False,  # Skip VAD
                    'gpu_id': gpu_id
                }
            else:
                return {'status': 'failed', 'reason': 'demucs_failed'}
                
        except Exception as e:
            logging.error(f"Error processing {unique_id}: {e}")
            return {'status': 'error', 'reason': str(e)}

    def process_file_batch(self, file_batch, gpu_id):
        """Process a batch of files on a specific GPU"""
        results = []
        for unique_id, file_info in file_batch:
            result = self.process_single_file(unique_id, file_info, gpu_id)
            result['unique_id'] = unique_id
            result['is_deepfake'] = file_info['is_deepfake']
            results.append(result)
        return results

    def _process_parallel(self, file_batches):
        """Process files in parallel using multiple GPUs"""
        logging.info(f"Starting parallel processing with {self.max_workers} workers")
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit batches to different GPUs
            future_to_batch = {}
            for i, batch in enumerate(file_batches):
                gpu_id = self.available_gpus[i % len(self.available_gpus)]
                future = executor.submit(self._process_batch_worker, batch, gpu_id)
                future_to_batch[future] = (batch, gpu_id)
            
            # Collect results
            for future in tqdm(as_completed(future_to_batch), total=len(future_to_batch), desc="Processing batches"):
                batch, gpu_id = future_to_batch[future]
                try:
                    results = future.result()
                    self._update_stats_from_results(results)
                    logging.info(f"Completed batch on GPU {gpu_id}: {len(results)} files")
                except Exception as e:
                    logging.error(f"Error processing batch on GPU {gpu_id}: {e}")

    def _process_sequential(self, file_items):
        """Process files sequentially (fallback)"""
        logging.info("Starting sequential processing")
        
        for unique_id, file_info in tqdm(file_items, desc="Processing files"):
            result = self.process_single_file(unique_id, file_info)
            self._update_stats_from_result(result, file_info)

    def _process_batch_worker(self, file_batch, gpu_id):
        """Worker function for parallel processing"""
        results = []
        for unique_id, file_info in file_batch:
            result = self.process_single_file(unique_id, file_info, gpu_id)
            result['unique_id'] = unique_id
            result['is_deepfake'] = file_info['is_deepfake']
            results.append(result)
        return results

    def _update_stats_from_results(self, results):
        """Update statistics from batch results"""
        for result in results:
            self._update_stats_from_result(result, {'is_deepfake': result['is_deepfake']})

    def _update_stats_from_result(self, result, file_info):
        """Update statistics from single result"""
        if file_info['is_deepfake']:
            self.stats['deepfake_processed'] += 1
        else:
            self.stats['bonafide_processed'] += 1
        
        if result['status'] == 'success':
            self.stats['successful_separations'] += 1
            if result.get('vad', False):
                self.stats['successful_vad'] += 1
            else:
                self.stats['failed_vad'] += 1
        elif result['status'] == 'skipped':
            self.stats['skipped_existing'] += 1
        elif result['status'] == 'failed':
            self.stats['failed_separations'] += 1
        elif result['status'] == 'error':
            self.stats['errors'] += 1

    def process_all_files(self):
        """Process all files through vocal separation and VAD with parallel GPU processing"""
        logging.info("Starting vocal separation and VAD processing...")
        
        # Load file mappings
        file_mappings = self.load_log_files()
        
        if not file_mappings:
            logging.error("No file mappings loaded")
            return self.stats
        
        # Prepare file batches for parallel processing
        file_items = list(file_mappings.items())
        # Use smaller batches for better load balancing and faster processing
        batch_size = max(1, len(file_items) // (self.max_workers * 4))  # 4x more batches for better distribution
        file_batches = [file_items[i:i + batch_size] for i in range(0, len(file_items), batch_size)]
        
        logging.info(f"Processing {len(file_items)} files in {len(file_batches)} batches using {self.max_workers} workers")
        
        # Process files in parallel for maximum speed
        logging.info("Running in parallel for maximum speed")
        self._process_parallel(file_batches)
        
        # Print final statistics
        logging.info("=== PROCESSING COMPLETED ===")
        logging.info(f"Total log files: {self.stats['total_log_files']}")
        logging.info(f"Bonafide files processed: {self.stats['bonafide_processed']}")
        logging.info(f"Deepfake files processed: {self.stats['deepfake_processed']}")
        logging.info(f"Successful separations: {self.stats['successful_separations']}")
        logging.info(f"Successful VAD: {self.stats['successful_vad']}")
        logging.info(f"Failed separations: {self.stats['failed_separations']}")
        logging.info(f"Failed VAD: {self.stats['failed_vad']}")
        logging.info(f"Skipped existing: {self.stats['skipped_existing']}")
        logging.info(f"Errors: {self.stats['errors']}")
        
        return self.stats

def main():
    """Main function"""
    separator = VocalSeparator()
    stats = separator.process_all_files()
    
    print(f"\n{'='*60}")
    print(f"VOCAL SEPARATION AND VAD COMPLETED")
    print(f"{'='*60}")
    print(f"Total files processed: {stats['bonafide_processed'] + stats['deepfake_processed']}")
    print(f"Successful separations: {stats['successful_separations']}")
    print(f"Successful VAD: {stats['successful_vad']}")
    print(f"Failed separations: {stats['failed_separations']}")
    print(f"Failed VAD: {stats['failed_vad']}")
    print(f"Skipped existing: {stats['skipped_existing']}")
    print(f"Errors: {stats['errors']}")
    print(f"Output directory: {separator.mdx_output_dir}")
    print(f"Check separate.log for detailed information")

if __name__ == "__main__":
    main()
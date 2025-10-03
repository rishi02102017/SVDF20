#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone VAD Processing Script for SVDF-20 Dataset

This script processes Voice Activity Detection (VAD) on already-separated vocals.wav files
from Demucs output, following the exact SingFake methodology.

PROCESS:
1. Scan dataset/processed/mdx_extra/ for vocals.wav files
2. Run PyAnnote VAD on each vocals.wav file
3. Generate vocals.vad files with timestamps
4. Monitor for new Demucs outputs and process them in real-time
5. Skip files that already have .vad files

OUTPUT:
- vocals.vad files in the same folders as vocals.wav
- VAD timestamps in SingFake format: [HH:MM:SS.mmm --> HH:MM:SS.mmm] 0 SPEECH

Based on SingFake paper methodology:
- PyAnnote VAD with 3.0s minimum duration
- Same hyperparameters as SingFake
- Real-time processing of new Demucs outputs
"""

import os
import sys
import logging
import time
from pathlib import Path
from tqdm import tqdm
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# PyAnnote imports
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection

# Import configuration
from config import (
    PYANNOTE_AUTH_TOKEN, VAD_HYPERPARAMETERS, BASE_DIR
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(str(Path(BASE_DIR) / "dataset/logs/processing/vad_only.log")),
        logging.StreamHandler()
    ]
)

class VADProcessor:
    def __init__(self):
        self.base_dir = Path(BASE_DIR)
        self.processed_dir = self.base_dir / "dataset/processed/mdx_extra"
        
        # PyAnnote setup
        self.authtoken = PYANNOTE_AUTH_TOKEN
        if self.authtoken is None:
            logging.error("No PyAnnote auth token provided. Please set PYANNOTE_AUTH_TOKEN in config.py")
            sys.exit(1)
        
        # Initialize VAD pipeline
        self._setup_vad()
        
        # Statistics
        self.stats = {
            'total_vocals_found': 0,
            'vad_files_created': 0,
            'vad_files_skipped': 0,
            'vad_errors': 0,
            'processing_time': 0
        }
        
        logging.info(f"Initialized VAD Processor")
        logging.info(f"Processed directory: {self.processed_dir}")
        logging.info(f"PyAnnote auth token: {'Set' if self.authtoken else 'Not set'}")

    def _setup_vad(self):
        """Setup PyAnnote VAD pipeline using SingFake hyperparameters"""
        try:
            model = Model.from_pretrained("pyannote/segmentation", use_auth_token=self.authtoken)
            self.pipeline = VoiceActivityDetection(segmentation=model)
            
            # SingFake hyperparameters from config
            self.pipeline.instantiate(VAD_HYPERPARAMETERS)
            logging.info("PyAnnote VAD pipeline initialized successfully")
            logging.info(f"VAD hyperparameters: {VAD_HYPERPARAMETERS}")
        except Exception as e:
            logging.error(f"Failed to initialize PyAnnote VAD: {e}")
            sys.exit(1)

    def find_vocals_files(self):
        """Find all vocals.wav files in the processed directory"""
        vocals_files = []
        
        if not self.processed_dir.exists():
            logging.error(f"Processed directory does not exist: {self.processed_dir}")
            return vocals_files
        
        # Use simple directory iteration to avoid rglob issues
        for subdir in self.processed_dir.iterdir():
            if subdir.is_dir():
                vocals_file = subdir / "vocals.wav"
                if vocals_file.exists():
                    vocals_files.append(vocals_file)
        
        logging.info(f"Found {len(vocals_files)} vocals.wav files")
        return vocals_files

    def process_single_vad(self, vocals_file):
        """Process VAD for a single vocals.wav file"""
        try:
            # Check if .vad file already exists
            vad_file = vocals_file.with_suffix('.vad')
            if vad_file.exists():
                return {'status': 'skipped', 'reason': 'vad_exists', 'file': vocals_file.name}
            
            # Run VAD
            vad_result = self.pipeline(str(vocals_file))
            vad_text = str(vad_result)
            
            # Save VAD file
            with open(vad_file, 'w') as f:
                f.write(vad_text)
            
            logging.debug(f"VAD completed: {vocals_file.name}")
            return {'status': 'success', 'file': vocals_file.name, 'vad_file': vad_file.name}
            
        except Exception as e:
            logging.error(f"VAD failed for {vocals_file.name}: {e}")
            return {'status': 'error', 'reason': str(e), 'file': vocals_file.name}

    def process_vocals_batch(self, vocals_files, max_workers=None):
        """Process a batch of vocals files sequentially (VAD can't be pickled for multiprocessing)"""
        logging.info(f"Processing {len(vocals_files)} vocals files sequentially (VAD requires single-threaded processing)")
        
        start_time = time.time()
        
        # Process files sequentially due to PyAnnote pickling issues
        with tqdm(total=len(vocals_files), desc="Processing VAD") as pbar:
            for vocals_file in vocals_files:
                try:
                    result = self.process_single_vad(vocals_file)
                    
                    if result['status'] == 'success':
                        self.stats['vad_files_created'] += 1
                    elif result['status'] == 'skipped':
                        self.stats['vad_files_skipped'] += 1
                    elif result['status'] == 'error':
                        self.stats['vad_errors'] += 1
                        
                except Exception as e:
                    logging.error(f"Error processing {vocals_file.name}: {e}")
                    self.stats['vad_errors'] += 1
                
                pbar.update(1)
        
        self.stats['processing_time'] = time.time() - start_time
        self.stats['total_vocals_found'] = len(vocals_files)

    def process_existing_files(self, max_workers=None):
        """Process all existing vocals.wav files"""
        logging.info("Starting VAD processing for existing vocals.wav files...")
        
        vocals_files = self.find_vocals_files()
        if not vocals_files:
            logging.warning("No vocals.wav files found")
            return
        
        self.process_vocals_batch(vocals_files, max_workers)
        self._log_statistics()

    def _log_statistics(self):
        """Log processing statistics"""
        logging.info("="*60)
        logging.info("VAD PROCESSING STATISTICS")
        logging.info("="*60)
        logging.info(f"Total vocals files found: {self.stats['total_vocals_found']}")
        logging.info(f"VAD files created: {self.stats['vad_files_created']}")
        logging.info(f"VAD files skipped (already exist): {self.stats['vad_files_skipped']}")
        logging.info(f"VAD errors: {self.stats['vad_errors']}")
        logging.info(f"Processing time: {self.stats['processing_time']:.2f} seconds")
        
        if self.stats['total_vocals_found'] > 0:
            success_rate = (self.stats['vad_files_created'] / self.stats['total_vocals_found']) * 100
            logging.info(f"Success rate: {success_rate:.1f}%")

class NewFileHandler(FileSystemEventHandler):
    """Handler for monitoring new vocals.wav files"""
    
    def __init__(self, vad_processor):
        self.vad_processor = vad_processor
        self.pending_files = set()
        self.lock = threading.Lock()
    
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('vocals.wav'):
            with self.lock:
                self.pending_files.add(event.src_path)
                logging.info(f"New vocals.wav detected: {Path(event.src_path).name}")
    
    def on_moved(self, event):
        if not event.is_directory and event.dest_path.endswith('vocals.wav'):
            with self.lock:
                self.pending_files.add(event.dest_path)
                logging.info(f"New vocals.wav detected: {Path(event.dest_path).name}")
    
    def get_pending_files(self):
        """Get and clear pending files"""
        with self.lock:
            files = list(self.pending_files)
            self.pending_files.clear()
            return files

def monitor_and_process(vad_processor, max_workers=None):
    """Monitor for new files and process them in real-time"""
    logging.info("Starting real-time monitoring for new vocals.wav files...")
    
    # Setup file system monitoring
    event_handler = NewFileHandler(vad_processor)
    observer = Observer()
    observer.schedule(event_handler, str(vad_processor.processed_dir), recursive=True)
    observer.start()
    
    try:
        while True:
            # Check for new files every 10 seconds
            time.sleep(10)
            
            pending_files = event_handler.get_pending_files()
            if pending_files:
                # Convert to Path objects and filter existing files
                vocals_files = [Path(f) for f in pending_files if Path(f).exists()]
                
                if vocals_files:
                    logging.info(f"Processing {len(vocals_files)} new vocals.wav files")
                    vad_processor.process_vocals_batch(vocals_files, max_workers)
                    vad_processor._log_statistics()
    
    except KeyboardInterrupt:
        logging.info("Monitoring stopped by user")
    finally:
        observer.stop()
        observer.join()

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Standalone VAD Processing for SVDF-20 Dataset')
    parser.add_argument('--mode', choices=['batch', 'monitor', 'both'], default='both',
                       help='Processing mode: batch (existing files), monitor (real-time), or both')
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: min(CPU cores, 8))')
    parser.add_argument('--max-files', type=int, default=None,
                       help='Maximum number of files to process (for testing)')
    
    args = parser.parse_args()
    
    # Initialize VAD processor
    vad_processor = VADProcessor()
    
    if args.mode in ['batch', 'both']:
        # Process existing files
        vocals_files = vad_processor.find_vocals_files()
        
        if args.max_files:
            vocals_files = vocals_files[:args.max_files]
            logging.info(f"Limited to {args.max_files} files for testing")
        
        if vocals_files:
            vad_processor.process_vocals_batch(vocals_files, args.workers)
            vad_processor._log_statistics()
        else:
            logging.warning("No vocals.wav files found to process")
    
    if args.mode in ['monitor', 'both']:
        # Start real-time monitoring
        monitor_and_process(vad_processor, args.workers)

if __name__ == "__main__":
    main()

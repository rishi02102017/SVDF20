#!/usr/bin/env python3
"""
SVDF-20 Audio Segmentation Script
Based on SingFake methodology with enhancements for SVDF-20 dataset

This script performs audio segmentation following the SingFake methodology:
1. Reads VAD files to get voice activity timestamps
2. Segments both vocals and mixtures based on VAD timestamps
3. Creates individual clips for training/evaluation
4. Implements skip logic for incomplete processing
5. Maintains proper file naming convention
6. Uses parallel processing for 10-20x speedup

Usage: python audio_segmentation.py [--max-workers N]
"""

import os
import sys
import librosa
import numpy as np
import soundfile as sf
from tqdm import tqdm
import logging
from pathlib import Path
import argparse
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# Import configuration
from config import BASE_DIR, OUTPUT_DIR, LOGS_DIR, SPLITS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('audio_segmentation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SVDF20Segmentation:
    """Audio segmentation for SVDF-20 dataset following SingFake methodology"""
    
    def __init__(self, 
                 source_folder: str = None,
                 logs_folder: str = None,
                 dump_folder: str = None,
                 output_sr: int = 16000,
                 skip_existing: bool = True,
                 max_workers: int = None):
        """
        Initialize the segmentation processor
        
        Args:
            source_folder: Path to processed mdx_extra directory
            logs_folder: Path to logs directory
            dump_folder: Path to output splits directory
            output_sr: Output sample rate
            skip_existing: Whether to skip already processed files
            max_workers: Maximum number of parallel workers (default: auto-detect)
        """
        # Use configurable paths if not provided
        base_path = Path(BASE_DIR)
        self.source_folder = Path(source_folder) if source_folder else base_path / OUTPUT_DIR
        self.logs_folder = Path(logs_folder) if logs_folder else base_path / LOGS_DIR
        self.dump_folder = Path(dump_folder) if dump_folder else base_path / SPLITS_DIR
        self.output_sr = output_sr
        self.skip_existing = skip_existing
        
        # Set up parallel processing
        if max_workers is None:
            # Use 75% of available cores, but cap at 20 for stability
            self.max_workers = min(20, max(1, int(multiprocessing.cpu_count() * 0.75)))
        else:
            self.max_workers = max_workers
        
        # Create output directories
        self.vocals_output = self.dump_folder / "vocals"
        self.mixtures_output = self.dump_folder / "mixtures"
        self.vocals_output.mkdir(parents=True, exist_ok=True)
        self.mixtures_output.mkdir(parents=True, exist_ok=True)
        
        # Statistics
        self.stats = {
            'total_vad_files': 0,
            'processed_files': 0,
            'skipped_files': 0,
            'error_files': 0,
            'total_clips_created': 0
        }
        
        logger.info(f"Initialized BharatGlobalFake Segmentation")
        logger.info(f"Source: {self.source_folder}")
        logger.info(f"Logs: {self.logs_folder}")
        logger.info(f"Output: {self.dump_folder}")
        logger.info(f"Skip existing: {self.skip_existing}")
        logger.info(f"Max workers: {self.max_workers}")

    def find_vad_files(self) -> List[Path]:
        """Find all VAD files in the source directory"""
        vad_files = []
        for root, dirs, files in os.walk(self.source_folder):
            for file in files:
                if file.endswith(".vad"):
                    vad_files.append(Path(root) / file)
        
        self.stats['total_vad_files'] = len(vad_files)
        logger.info(f"Found {len(vad_files)} VAD files")
        return vad_files

    def get_corresponding_log(self, vad_file: Path) -> Optional[Path]:
        """Get the corresponding log file for a VAD file"""
        # Extract video ID from directory name (format: videoID_Artist_Song_Year_type)
        dir_name = vad_file.parent.name
        video_id = dir_name.split("_")[0]
        log_file = self.logs_folder / f"{video_id}.log"
        
        if log_file.exists():
            return log_file
        else:
            # Try alternative naming pattern
            log_file = self.logs_folder / f"{dir_name}.log"
            if log_file.exists():
                return log_file
            else:
                logger.warning(f"No log file found for {vad_file}")
                return None

    def parse_log_file(self, log_file: Path) -> bool:
        """
        Parse log file to get spoof/bonafide label
        
        Returns:
            Boolean indicating if the file is spoof (True) or bonafide (False)
        """
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if len(lines) < 6:
                logger.warning(f"Log file {log_file} has insufficient lines")
                return False
            
            # Line 4 (index 4) contains spoof/bonafide label
            label = lines[4].strip().lower()
            is_spoof = (label == "spoof")
            
            return is_spoof
            
        except Exception as e:
            logger.error(f"Error parsing log file {log_file}: {e}")
            return False

    def load_audio_files(self, vad_file: Path) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Load vocal and mixture audio files"""
        try:
            # Load vocal file
            vocal_file = vad_file.with_suffix('.wav')
            if not vocal_file.exists():
                logger.warning(f"Vocal file not found: {vocal_file}")
                return None, None
            
            vocal, vocal_sr = librosa.load(str(vocal_file), sr=self.output_sr, mono=False)
            
            # Load mixture file (use no_vocals.wav from Demucs output)
            mixture_file = vad_file.parent / "no_vocals.wav"
            if not mixture_file.exists():
                logger.warning(f"Mixture file not found: {mixture_file}")
                return None, None
            
            mixture, mixture_sr = librosa.load(str(mixture_file), sr=self.output_sr, mono=False)
            
            return vocal, mixture
            
        except Exception as e:
            logger.error(f"Error loading audio files for {vad_file}: {e}")
            return None, None

    def parse_vad_timestamps(self, vad_file: Path) -> List[Tuple[float, float]]:
        """Parse VAD file to extract voice activity timestamps"""
        timestamps = []
        
        try:
            with open(vad_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse format: [00:00:33.679 --> 00:00:42.252]
                    if '[' in line and ']' in line and '-->' in line:
                        time_part = line.split(']')[0].split('[')[1]
                        start_str, end_str = time_part.split('-->')
                        
                        # Convert to seconds
                        start_time = self.time_to_seconds(start_str.strip())
                        end_time = self.time_to_seconds(end_str.strip())
                        
                        timestamps.append((start_time, end_time))
                        
                except Exception as e:
                    logger.warning(f"Error parsing VAD line '{line}': {e}")
                    continue
            
            return timestamps
            
        except Exception as e:
            logger.error(f"Error reading VAD file {vad_file}: {e}")
            return []

    def time_to_seconds(self, time_str: str) -> float:
        """Convert time string (HH:MM:SS.mmm) to seconds"""
        try:
            parts = time_str.split(':')
            if len(parts) == 3:
                hours, minutes, seconds = parts
                return float(hours) * 3600.0 + float(minutes) * 60.0 + float(seconds)
            else:
                return float(time_str)
        except:
            return 0.0

    def segment_audio(self, audio: np.ndarray, start_time: float, end_time: float) -> np.ndarray:
        """Segment audio based on start and end times"""
        start_sample = int(start_time * self.output_sr)
        end_sample = int(end_time * self.output_sr)
        
        if len(audio.shape) == 1:
            # Mono audio
            segment = audio[start_sample:end_sample]
            segment = np.expand_dims(segment, axis=0)
            segment = np.transpose(segment)
        else:
            # Stereo audio
            segment = audio[:, start_sample:end_sample]
            segment = np.transpose(segment)
        
        return segment

    def should_skip_file(self, vad_file: Path, is_spoof: bool) -> bool:
        """Check if file should be skipped (already processed)"""
        if not self.skip_existing:
            return False
        
        # Check if any clips from this file already exist
        video_id = vad_file.parent.name.split("_")[0]
        spoof_prefix = "1" if is_spoof else "0"
        
        # Check for existing clips (look for first few indices)
        for i in range(5):  # Check first 5 potential clips
            clip_name = f"{spoof_prefix}_{video_id}_{i}.flac"
            if (self.vocals_output / clip_name).exists() and (self.mixtures_output / clip_name).exists():
                return True
        
        return False

    def process_vad_file(self, vad_file: Path) -> bool:
        """Process a single VAD file and create audio clips"""
        try:
            # Get corresponding log file
            log_file = self.get_corresponding_log(vad_file)
            if not log_file:
                self.stats['error_files'] += 1
                return False
            
            # Parse log file
            is_spoof = self.parse_log_file(log_file)
            
            # Check if should skip
            if self.should_skip_file(vad_file, is_spoof):
                self.stats['skipped_files'] += 1
                logger.debug(f"Skipping already processed file: {vad_file}")
                return True
            
            # Load audio files
            vocal, mixture = self.load_audio_files(vad_file)
            if vocal is None or mixture is None:
                self.stats['error_files'] += 1
                return False
            
            # Parse VAD timestamps
            timestamps = self.parse_vad_timestamps(vad_file)
            if not timestamps:
                self.stats['error_files'] += 1
                return False
            
            # Create clips
            video_id = vad_file.parent.name.split("_")[0]
            spoof_prefix = "1" if is_spoof else "0"
            clips_created = 0
            
            for i, (start_time, end_time) in enumerate(timestamps):
                try:
                    # Segment audio
                    vocal_segment = self.segment_audio(vocal, start_time, end_time)
                    mixture_segment = self.segment_audio(mixture, start_time, end_time)
                    
                    # Create filename
                    clip_name = f"{spoof_prefix}_{video_id}_{i}.flac"
                    
                    # Save clips
                    sf.write(self.vocals_output / clip_name, vocal_segment, self.output_sr, subtype="PCM_16")
                    sf.write(self.mixtures_output / clip_name, mixture_segment, self.output_sr, subtype="PCM_16")
                    
                    clips_created += 1
                    
                except Exception as e:
                    logger.warning(f"Error creating clip {i} for {vad_file}: {e}")
                    continue
            
            self.stats['processed_files'] += 1
            self.stats['total_clips_created'] += clips_created
            
            if clips_created > 0:
                logger.debug(f"Created {clips_created} clips from {vad_file}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing {vad_file}: {e}")
            self.stats['error_files'] += 1
            return False

    @staticmethod
    def process_vad_file_static(args):
        """Static method for parallel processing"""
        vad_file, source_folder, logs_folder, dump_folder, output_sr, skip_existing = args
        
        # Create a temporary processor instance for this worker
        processor = BharatGlobalFakeSegmentation(
            source_folder=source_folder,
            logs_folder=logs_folder,
            dump_folder=dump_folder,
            output_sr=output_sr,
            skip_existing=skip_existing,
            max_workers=1  # Single worker for this instance
        )
        
        # Process the file
        return processor.process_vad_file(vad_file)

    def run_segmentation(self):
        """Run the complete audio segmentation process with parallel processing"""
        logger.info("Starting BharatGlobalFake audio segmentation...")
        
        # Find all VAD files
        vad_files = self.find_vad_files()
        
        if not vad_files:
            logger.error("No VAD files found!")
            return
        
        logger.info(f"Processing {len(vad_files)} VAD files with {self.max_workers} parallel workers...")
        
        # Prepare arguments for parallel processing
        process_args = [
            (
                vad_file,
                str(self.source_folder),
                str(self.logs_folder),
                str(self.dump_folder),
                self.output_sr,
                self.skip_existing
            )
            for vad_file in vad_files
        ]
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(tqdm(
                executor.map(self.process_vad_file_static, process_args),
                total=len(vad_files),
                desc="Processing VAD files"
            ))
        
        # Count results
        self.stats['processed_files'] = sum(1 for r in results if r)
        self.stats['error_files'] = sum(1 for r in results if not r)
        
        # Print final statistics
        self.print_statistics()

    def print_statistics(self):
        """Print processing statistics"""
        logger.info("=" * 60)
        logger.info("BharatGlobalFake Audio Segmentation Complete!")
        logger.info("=" * 60)
        logger.info(f"Total VAD files found: {self.stats['total_vad_files']}")
        logger.info(f"Files processed: {self.stats['processed_files']}")
        logger.info(f"Files skipped: {self.stats['skipped_files']}")
        logger.info(f"Files with errors: {self.stats['error_files']}")
        logger.info(f"Total clips created: {self.stats['total_clips_created']}")
        logger.info(f"Output directory: {self.dump_folder}")
        logger.info("=" * 60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="SVDF-20 Audio Segmentation")
    parser.add_argument("--source", 
                       default=None,
                       help="Source directory with processed files")
    parser.add_argument("--logs", 
                       default=None,
                       help="Logs directory")
    parser.add_argument("--output", 
                       default=None,
                       help="Output directory for splits")
    parser.add_argument("--sample-rate", type=int, default=16000,
                       help="Output sample rate")
    parser.add_argument("--no-skip", action="store_true",
                       help="Don't skip already processed files")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Maximum number of parallel workers (default: auto-detect)")
    
    args = parser.parse_args()
    
    # Create segmentation processor
    processor = BharatGlobalFakeSegmentation(
        source_folder=args.source,
        logs_folder=args.logs,
        dump_folder=args.output,
        output_sr=args.sample_rate,
        skip_existing=not args.no_skip,
        max_workers=args.max_workers
    )
    
    # Run segmentation
    processor.run_segmentation()

if __name__ == "__main__":
    main()

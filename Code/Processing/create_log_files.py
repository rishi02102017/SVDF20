#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create SingFake-compatible log files from SVDF-20 CSV tracking data

This script converts our CSV tracking files into the exact log file format
that SingFake processing scripts expect.

Log file format (8 lines) - SingFake compatible:
Line 1: {unique_id}                    # e.g., "Kt91AjgLsw4_Lata Mangeshkar_Aap Ki Nazron Ne Samjha_1962"
Line 2: {title}                        # e.g., "Aap Ki Nazron Ne Samjha"
Line 3: {url}                          # YouTube URL
Line 4: {singer}                       # e.g., "Lata Mangeshkar"
Line 5: {bonafide_or_spoof}            # "bonafide" or "spoof"
Line 6: {model}                        # AI model used (for spoofs) or "nan" (for bonafide)
Line 7: {language}                     # Language code
Line 8: {split}                        # "Training", "Validation", "T01", "T02", "T04"

Note: 
- NO PREFIXES (0_ or 1_) in filenames - these are added later during segmentation step
- T03 is NOT created here - it's generated later from T02 using codec simulation
- T03 = T02 processed through 4 communication codecs (MP3, AAC, OPUS, Vorbis)
"""

import os
import sys
import pandas as pd
import logging
from pathlib import Path
from tqdm import tqdm

# Import configuration
from config import BASE_DIR, LOGS_DIR
import re

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('dataset/logs/processing/create_log_files.log'),
        logging.StreamHandler()
    ]
)

class LogFileCreator:
    def __init__(self):
        self.base_dir = Path("/data-caffe/rishabh/SingFake_Project/IndicFake")
        
        # Input CSV files
        self.bonafide_csv = self.base_dir / "dataset/tracking/successful_downloads.csv"
        self.deepfake_csv = self.base_dir / "dataset/tracking_deepfake/deepfake_successful_downloads.csv"
        
        # Output directory
        self.logs_dir = self.base_dir / "dataset/logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Dataset splits (following SingFake methodology - T03 is created later from T02)
        self.splits = {
            'Training': 0.6035,    # 60.35%
            'Validation': 0.1008,  # 10.08%
            'T01': 0.0975,         # 9.75% (Seen singers)
            'T02': 0.1662,         # 16.62% (Unseen singers) - T03 will be created from this
            'T04': 0.0321          # 3.21% (Unseen contexts)
        }
        
        logging.info(f"Initialized LogFileCreator")
        logging.info(f"Bonafide CSV: {self.bonafide_csv}")
        logging.info(f"Deepfake CSV: {self.deepfake_csv}")
        logging.info(f"Logs directory: {self.logs_dir}")

    def sanitize_filename(self, text):
        """Create safe filename from text, preserving spaces to match audio filenames"""
        # Convert to string and handle None values
        text = str(text) if text is not None else ""
        
        # Replace forward slashes and other problematic characters
        safe_name = text.replace('/', '_').replace('\\', '_').replace(':', '_')
        safe_name = safe_name.replace('*', '_').replace('?', '_').replace('"', '_')
        safe_name = safe_name.replace('<', '_').replace('>', '_').replace('|', '_')
        
        # Remove other special characters but keep Unicode characters
        safe_name = re.sub(r'[^\w\s\-_.]', '_', safe_name)
        
        # Replace multiple dashes/underscores with single underscore, but preserve spaces
        safe_name = re.sub(r'[-_]+', '_', safe_name)
        
        # Remove leading/trailing underscores
        safe_name = safe_name.strip('_')
        
        # Limit length to avoid filesystem issues
        safe_name = safe_name[:100]  # Increased from 50 to handle longer names
        
        return safe_name

    def extract_youtube_id(self, url):
        """Extract YouTube ID from URL"""
        try:
            # Handle different YouTube URL formats
            if 'youtube.com/watch?v=' in url:
                return url.split('v=')[1].split('&')[0]
            elif 'youtu.be/' in url:
                return url.split('youtu.be/')[1].split('?')[0]
            else:
                return None
        except Exception:
            return None

    def create_unique_id(self, singer, title, year, youtube_id, is_deepfake=False):
        """Create unique identifier for log file following SingFake convention (NO PREFIXES)"""
        # SingFake convention: {youtube_id}_{singer}_{title}_{year} (no 0_ or 1_ prefixes)
        # Prefixes are added later during segmentation step
        # Sanitize singer and title to handle special characters
        safe_singer = self.sanitize_filename(singer)
        safe_title = self.sanitize_filename(title)
        
        # Handle special case where YouTube ID might be None (for unknown_ files)
        if youtube_id:
            base_id = f"{youtube_id}_{safe_singer}_{safe_title}_{year}"
        else:
            base_id = f"unknown_{safe_singer}_{safe_title}_{year}"
        
        # Add suffix to distinguish bonafide vs deepfake versions of the same song
        if is_deepfake:
            return f"{base_id}_deepfake"
        else:
            return f"{base_id}_bonafide"

    def assign_split(self, index, total_count, is_bonafide):
        """Assign dataset split based on index and type"""
        # Use different random seeds for bonafide vs spoof to ensure balanced splits
        import random
        random.seed(42 + (1 if is_bonafide else 0))
        
        # Shuffle indices
        indices = list(range(total_count))
        random.shuffle(indices)
        
        # Find position of current index in shuffled list
        shuffled_pos = indices.index(index)
        
        # Assign split based on position (matching SingFake methodology)
        cumulative = 0
        for split_name, ratio in self.splits.items():
            cumulative += ratio
            if shuffled_pos < total_count * cumulative:
                return split_name
        # Fallback to last split if rounding errors occur
        return list(self.splits.keys())[-1]

    def create_log_file(self, unique_id, title, url, singer, label, model, language, split):
        """Create a single log file in SingFake format"""
        try:
            log_file = self.logs_dir / f"{unique_id}.log"
            
            # Skip if file already exists
            if log_file.exists():
                logging.debug(f"Log file already exists, skipping: {unique_id}")
                return True
            
            with open(log_file, 'w', encoding='utf-8') as f:
                f.write(f"{unique_id}\n")           # Line 1: filename
                f.write(f"{title}\n")               # Line 2: title
                f.write(f"{url}\n")                 # Line 3: url
                f.write(f"{singer}\n")              # Line 4: singer
                f.write(f"{label}\n")               # Line 5: bonafide/spoof
                f.write(f"{model}\n")               # Line 6: model
                f.write(f"{language}\n")            # Line 7: language
                f.write(f"{split}\n")               # Line 8: split
            
            return True
        except Exception as e:
            logging.error(f"Error creating log file for {unique_id}: {e}")
            return False

    def process_bonafide_data(self):
        """Process bonafide songs and create log files"""
        logging.info("Processing bonafide songs...")
        
        if not self.bonafide_csv.exists():
            logging.error(f"Bonafide CSV not found: {self.bonafide_csv}")
            return 0
        
        # Load bonafide data
        df_bonafide = pd.read_csv(self.bonafide_csv)
        logging.info(f"Loaded {len(df_bonafide)} bonafide songs")
        
        # Remove duplicates to prevent overwrites
        df_bonafide_clean = df_bonafide.drop_duplicates(subset=['singer', 'title', 'year', 'url_used'], keep='first')
        duplicates_removed = len(df_bonafide) - len(df_bonafide_clean)
        if duplicates_removed > 0:
            logging.info(f"Removed {duplicates_removed} duplicate bonafide entries")
        
        # Reset index to avoid indexing issues after duplicate removal
        df_bonafide_clean = df_bonafide_clean.reset_index(drop=True)
        
        created_count = 0
        skipped_count = 0
        
        for idx, row in tqdm(df_bonafide_clean.iterrows(), total=len(df_bonafide_clean), desc="Creating bonafide logs"):
            try:
                # Extract data
                singer = row['singer']
                title = row['title']
                year = row['year']
                language = row['Language']
                url = row['url_used']
                
                # Extract YouTube ID from URL
                youtube_id = self.extract_youtube_id(url)
                if not youtube_id:
                    logging.warning(f"Could not extract YouTube ID from URL: {url}, using 'unknown' prefix")
                    youtube_id = None  # Will be handled in create_unique_id
                
                # Create unique ID following SingFake convention (no prefixes)
                unique_id = self.create_unique_id(singer, title, year, youtube_id, is_deepfake=False)
                
                # Assign split
                split = self.assign_split(idx, len(df_bonafide_clean), is_bonafide=True)
                
                # Check if file already exists before processing
                log_file = self.logs_dir / f"{unique_id}.log"
                if log_file.exists():
                    skipped_count += 1
                    continue
                
                # Create log file
                success = self.create_log_file(
                    unique_id=unique_id,
                    title=title,
                    url=url,
                    singer=singer,
                    label="bonafide",
                    model="nan",
                    language=language,
                    split=split
                )
                
                if success:
                    created_count += 1
                    
            except Exception as e:
                logging.error(f"Error processing bonafide row {idx}: {e}")
                continue
        
        logging.info(f"Created {created_count} bonafide log files, skipped {skipped_count} existing files")
        return created_count

    def process_deepfake_data(self):
        """Process deepfake songs and create log files"""
        logging.info("Processing deepfake songs...")
        
        if not self.deepfake_csv.exists():
            logging.error(f"Deepfake CSV not found: {self.deepfake_csv}")
            return 0
        
        # Load deepfake data
        df_deepfake = pd.read_csv(self.deepfake_csv)
        logging.info(f"Loaded {len(df_deepfake)} deepfake songs")
        
        # Remove duplicates to prevent overwrites
        df_deepfake_clean = df_deepfake.drop_duplicates(subset=['singer', 'title', 'year', 'youtube_url'], keep='first')
        duplicates_removed = len(df_deepfake) - len(df_deepfake_clean)
        if duplicates_removed > 0:
            logging.info(f"Removed {duplicates_removed} duplicate deepfake entries")
        
        # Reset index to avoid indexing issues after duplicate removal
        df_deepfake_clean = df_deepfake_clean.reset_index(drop=True)
        
        created_count = 0
        skipped_count = 0
        
        for idx, row in tqdm(df_deepfake_clean.iterrows(), total=len(df_deepfake_clean), desc="Creating deepfake logs"):
            try:
                # Extract data
                singer = row['singer']
                title = row['title']
                year = row['year']
                language = row['Language']
                url = row['youtube_url']
                model = row.get('ai_model_detected', 'Unknown')
                youtube_id = row.get('youtube_id', '')
                
                # Use YouTube ID from CSV or extract from URL
                if not youtube_id:
                    youtube_id = self.extract_youtube_id(url)
                    if not youtube_id:
                        logging.warning(f"Could not extract YouTube ID from URL: {url}")
                        continue
                
                # Create unique ID following SingFake convention (no prefixes)
                unique_id = self.create_unique_id(singer, title, year, youtube_id, is_deepfake=True)
                
                # Assign split
                split = self.assign_split(idx, len(df_deepfake_clean), is_bonafide=False)
                
                # Check if file already exists before processing
                log_file = self.logs_dir / f"{unique_id}.log"
                if log_file.exists():
                    skipped_count += 1
                    continue
                
                # Create log file
                success = self.create_log_file(
                    unique_id=unique_id,
                    title=title,
                    url=url,
                    singer=singer,
                    label="spoof",
                    model=model,
                    language=language,
                    split=split
                )
                
                if success:
                    created_count += 1
                    
            except Exception as e:
                logging.error(f"Error processing deepfake row {idx}: {e}")
                continue
        
        logging.info(f"Created {created_count} deepfake log files, skipped {skipped_count} existing files")
        return created_count

    def validate_log_files(self):
        """Validate created log files"""
        logging.info("Validating log files...")
        
        log_files = list(self.logs_dir.glob("*.log"))
        logging.info(f"Found {len(log_files)} log files")
        
        # Count by split and label
        split_counts = {}
        label_counts = {}
        
        for log_file in log_files:
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if len(lines) >= 8:
                        label = lines[4].strip()
                        split = lines[7].strip()
                        
                        # Count labels
                        label_counts[label] = label_counts.get(label, 0) + 1
                        
                        # Count splits
                        if split not in split_counts:
                            split_counts[split] = {}
                        split_counts[split][label] = split_counts[split].get(label, 0) + 1
                        
            except Exception as e:
                logging.error(f"Error reading log file {log_file}: {e}")
                continue
        
        # Report statistics
        logging.info("Label distribution:")
        for label, count in label_counts.items():
            logging.info(f"  {label}: {count}")
        
        logging.info("Split distribution:")
        for split, labels in split_counts.items():
            logging.info(f"  {split}:")
            for label, count in labels.items():
                logging.info(f"    {label}: {count}")
        
        return split_counts, label_counts

    def count_existing_log_files(self):
        """Count existing log files to show progress"""
        existing_files = list(self.logs_dir.glob("*.log"))
        existing_count = len(existing_files)
        logging.info(f"Found {existing_count} existing log files")
        return existing_count

    def identify_missing_files(self):
        """Identify which specific files are missing (for debugging)"""
        import pandas as pd
        
        # Load clean data
        df_bonafide = pd.read_csv(self.bonafide_csv)
        df_bonafide_clean = df_bonafide.drop_duplicates(subset=['singer', 'title', 'year', 'url_used'], keep='first')
        df_bonafide_clean = df_bonafide_clean.reset_index(drop=True)
        
        df_deepfake = pd.read_csv(self.deepfake_csv)
        df_deepfake_clean = df_deepfake.drop_duplicates(subset=['singer', 'title', 'year', 'youtube_url'], keep='first')
        df_deepfake_clean = df_deepfake_clean.reset_index(drop=True)
        
        expected_total = len(df_bonafide_clean) + len(df_deepfake_clean)
        existing_count = self.count_existing_log_files()
        missing_count = expected_total - existing_count
        
        if missing_count > 0:
            logging.info(f"Missing {missing_count} log files out of {expected_total} expected")
        else:
            logging.info(f"All {expected_total} log files are present")
        
        return missing_count

    def create_all_log_files(self):
        """Create all log files"""
        logging.info("Starting log file creation...")
        
        # Count existing files and identify missing ones
        existing_count = self.count_existing_log_files()
        missing_count = self.identify_missing_files()
        
        if missing_count == 0:
            logging.info("All log files already exist! Nothing to do.")
            return existing_count, 0, {}, {}
        
        # Process bonafide songs
        bonafide_count = self.process_bonafide_data()
        
        # Process deepfake songs
        deepfake_count = self.process_deepfake_data()
        
        # Validate results
        split_counts, label_counts = self.validate_log_files()
        
        logging.info(f"Log file creation completed!")
        logging.info(f"Total bonafide logs: {bonafide_count}")
        logging.info(f"Total deepfake logs: {deepfake_count}")
        logging.info(f"Total log files: {bonafide_count + deepfake_count}")
        logging.info(f"Files created in this run: {bonafide_count + deepfake_count - existing_count}")
        
        return bonafide_count, deepfake_count, split_counts, label_counts

def main():
    """Main function"""
    creator = LogFileCreator()
    bonafide_count, deepfake_count, split_counts, label_counts = creator.create_all_log_files()
    
    print(f"\n{'='*50}")
    print(f"LOG FILE CREATION COMPLETED")
    print(f"{'='*50}")
    print(f"Bonafide logs: {bonafide_count}")
    print(f"Deepfake logs: {deepfake_count}")
    print(f"Total logs: {bonafide_count + deepfake_count}")
    print(f"Logs directory: {creator.logs_dir}")
    print(f"Check create_log_files.log for detailed information")

if __name__ == "__main__":
    main()

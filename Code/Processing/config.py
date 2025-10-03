#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Configuration file for SVDF-20 processing pipeline

This file contains configuration settings for the processing scripts.
Users should modify these paths according to their setup.
"""

import os

# PyAnnote Configuration
# Get your auth token from: https://huggingface.co/settings/tokens
# Users must set this token to use PyAnnote VAD functionality
PYANNOTE_AUTH_TOKEN = os.getenv("PYANNOTE_AUTH_TOKEN", "YOUR_HUGGINGFACE_TOKEN_HERE")

# Demucs Configuration
DEMUCS_MODEL = "mdx_extra"  # SingFake uses mdx_extra model
DEMUCS_STEMS = "vocals"     # Extract vocals and instrumental

# VAD Configuration (SingFake hyperparameters)
VAD_HYPERPARAMETERS = {
    "onset": 0.5,           # Onset activation threshold
    "offset": 0.5,          # Offset activation threshold
    "min_duration_on": 3.0, # Remove speech regions shorter than 3 seconds
    "min_duration_off": 0.0 # Fill non-speech regions shorter than 0 seconds
}

# Audio Processing Configuration
OUTPUT_SAMPLE_RATE = 16000  # 16kHz as per SingFake paper
OUTPUT_FORMAT = "wav"       # Output format for separated audio

# Paths Configuration - Users should modify these according to their setup
BASE_DIR = os.getenv("SVDF_BASE_DIR", "/path/to/your/svdf/dataset")
LOGS_DIR = os.getenv("SVDF_LOGS_DIR", "dataset/logs")
BONAFIDE_DIR = os.getenv("SVDF_BONAFIDE_DIR", "dataset/raw_downloads")
DEEPFAKE_DIR = os.getenv("SVDF_DEEPFAKE_DIR", "dataset/raw_downloads_deepfake")
OUTPUT_DIR = os.getenv("SVDF_OUTPUT_DIR", "dataset/processed/mdx_extra")
SPLITS_DIR = os.getenv("SVDF_SPLITS_DIR", "dataset/splits")

# Processing Configuration
TIMEOUT_SECONDS = 300       # Timeout for VAD processing
MAX_RETRIES = 3            # Maximum retries for failed operations
BATCH_SIZE = 1             # Process files one at a time (Demucs limitation)

# GPU Configuration
USE_GPU = True             # Enable GPU acceleration
MAX_GPU_WORKERS = 4        # Maximum number of parallel GPU workers
GPU_MEMORY_FRACTION = 0.8  # Fraction of GPU memory to use (0.8 = 80%)

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
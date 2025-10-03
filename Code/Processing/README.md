# SVDF-20 Processing Pipeline

This directory contains the processing scripts for the SVDF-20 dataset, following the SingFake methodology for singing voice deepfake detection.

## Overview

The processing pipeline consists of the following steps:

1. **Log File Creation** (`create_log_files.py`) - Convert CSV tracking data to SingFake-compatible log files
2. **Vocal Separation** (`separate.py`) - Extract vocals using Demucs (VAD skipped for speed)
3. **Voice Activity Detection** (`vad_only.py`) - Perform VAD on separated vocals
4. **Audio Segmentation** (`audio_segmentation.py`) - Segment audio into training clips based on VAD timestamps
5. **Dataset Splitting** (`create_dataset_splits.py`) - Create train/validation/test splits following SingFake methodology

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended for Demucs)
- HuggingFace account for PyAnnote access

## Installation

```bash
pip install demucs pyannote.audio librosa soundfile tqdm pandas numpy torch torchaudio
```

## Configuration

1. **Set up environment variables** (modify according to your setup):
```bash
export SVDF_BASE_DIR="/path/to/your/svdf/dataset"
export SVDF_LOGS_DIR="dataset/logs"
export SVDF_BONAFIDE_DIR="dataset/raw_downloads"
export SVDF_DEEPFAKE_DIR="dataset/raw_downloads_deepfake"
export SVDF_OUTPUT_DIR="dataset/processed/mdx_extra"
export SVDF_SPLITS_DIR="dataset/splits"
```

2. **Set HuggingFace token** for PyAnnote:
```bash
export PYANNOTE_AUTH_TOKEN="your_huggingface_token_here"
```

3. **Modify `config.py`** if needed for your specific setup.

## Usage

### Step 1: Create Log Files
```bash
python create_log_files.py
```

### Step 2: Vocal Separation
```bash
python separate.py
```

### Step 3: Voice Activity Detection
```bash
python vad_only.py
```

### Step 4: Audio Segmentation
```bash
python audio_segmentation.py
```

### Step 5: Dataset Splitting
```bash
python create_dataset_splits.py
```

## Directory Structure

```
svdf_dataset/
├── dataset/
│   ├── logs/                    # Log files (created in Step 1)
│   ├── raw_downloads/           # Bonafide audio files
│   ├── raw_downloads_deepfake/  # Deepfake audio files
│   ├── processed/               # Separated audio (created in Step 2)
│   └── splits/                  # Final clips (created in Step 3)
└── scripts/
    └── processing_anonymous/
        ├── config.py
        ├── separate.py
        ├── audio_segmentation.py
        └── create_dataset_splits.py
```

## Notes

- **GPU Acceleration**: Demucs processing is significantly faster with GPU. Adjust `MAX_GPU_WORKERS` in `config.py` based on your hardware.
- **Memory Requirements**: Processing the full dataset requires substantial storage and memory.
- **Processing Time**: Full pipeline may take several hours depending on hardware.
- **Quality Control**: The pipeline includes automatic quality filtering as described in the paper.

## Troubleshooting

- **PyAnnote authentication**: Ensure your HuggingFace token is valid and has access to PyAnnote models
- **CUDA issues**: The pipeline will fall back to CPU if GPU is unavailable
- **Memory errors**: Reduce `BATCH_SIZE` or `MAX_GPU_WORKERS` in `config.py`
- **File not found**: Verify your directory structure matches the expected layout

## References

- SingFake paper methodology
- Demucs: https://github.com/facebookresearch/demucs
- PyAnnote: https://github.com/pyannote/pyannote-audio
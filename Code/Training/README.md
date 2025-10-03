# SVDF-20 Training Code

This repository contains the training code for the SVDF-20 dataset as described in our paper "THE SVDF-20 BENCHMARK: BREAKING LANGUAGE BARRIERS IN SINGING VOICE DEEPFAKE DETECTION".

## Overview

This codebase supports training 8 state-of-the-art audio spoofing detection models on the SVDF-20 multilingual dataset:

- **AASIST** - Anti-spoofing with Integrated Spectro-Temporal Graph Attention Networks
- **RawGAT_ST** - Raw waveform with Graph Attention Networks  
- **RawNet2** - Raw waveform neural networks
- **SpecRNet** - Spectral Residual Networks
- **Whisper** - OpenAI Whisper-based detection
- **SSLModel** - Self-supervised learning models
- **Conformer** - Convolution-augmented Transformer
- **RawNetLite** - Lightweight RawNet variant

## Requirements

- Python 3.8+
- CUDA-capable GPU (recommended)
- 32GB+ GPU memory for full training

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd training_anonymous

# Install dependencies
pip install -r requirements.txt

# Download XLSR model (for SSLModel and Conformer)
wget https://huggingface.co/facebook/wav2vec2-xlsr-300m/resolve/main/pytorch_model.bin -O src/models/xlsr2_300m.pt
```

## Dataset Setup

1. **Download the SVDF-20 dataset** (not included in this repository)
2. **Set environment variable**:
   ```bash
   export SVDF_DATASET_PATH="/path/to/your/svdf/dataset"
   ```

3. **Expected dataset structure**:
   ```
   dataset/
   ├── final_splits/
   │   ├── Training/
   │   │   ├── vocals/
   │   │   └── no_vocals/
   │   ├── Validation/
   │   │   ├── vocals/
   │   │   └── no_vocals/
   │   ├── T01/
   │   │   ├── vocals/
   │   │   └── no_vocals/
   │   ├── T02/
   │   │   ├── vocals/
   │   │   └── no_vocals/
   │   ├── T03/
   │   │   ├── vocals/
   │   │   └── no_vocals/
   │   └── T04/
   │       ├── vocals/
   │       └── no_vocals/
   ```

## Usage

### Train Individual Model

```bash
python train.py --model_name AASIST --batch_size 32 --num_epochs 25
```

### Train All Models

```bash
python train_all.py
```

### Custom Training

```bash
python train.py \
    --model_name RawNet2 \
    --batch_size 16 \
    --num_epochs 30 \
    --lr 1e-4 \
    --database_path /path/to/dataset
```

## Configuration

Edit `config.py` to modify default parameters:

- `--database_path`: Path to SVDF-20 dataset
- `--model_name`: Model to train  
- `--batch_size`: Batch size (adjust for GPU memory)
- `--num_epochs`: Number of training epochs (default: 25)
- `--lr`: Learning rate (default: 3e-4)
- `--xlsr_model_path`: Path to XLSR model for SSL models

## Training Details

- **Epochs**: 25 (as reported in paper)
- **Batch Size**: 32 (optimized for V100 32GB GPUs)
- **Learning Rate**: 3e-4
- **Optimizer**: Adam with weight decay
- **Mixed Precision**: FP16 for faster training
- **Class Weighting**: Handles bonafide/deepfake imbalance

## Output

Training outputs:
- Model checkpoints in `checkpoints/`
- Training logs in `logs/`
- Evaluation results in `results/`

## Reproducing Results

To reproduce the results from our paper:

1. **Download the complete SVDF-20 dataset**
2. **Set up the dataset structure as shown above**
3. **Run training with default parameters**:
   ```bash
   python train_all.py
   ```
4. **Evaluate on test sets** using the provided evaluation scripts

## Citation

```bibtex
@article{svdf20_2024,
  title={THE SVDF-20 BENCHMARK: BREAKING LANGUAGE BARRIERS IN SINGING VOICE DEEPFAKE DETECTION},
  author={Anonymous},
  journal={ICLR},
  year={2024}
}
```

## License

This code is released for research purposes. Please cite our paper if you use this code in your research.

## Notes

- Training on the full SVDF-20 dataset requires substantial computational resources (~4 GPU-days per model)
- Models are trained on multilingual data (20 languages)
- Training follows the SingFake methodology with singer-based splitting
- RawNetLite is the newest addition with improved efficiency
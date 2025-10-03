# SVDF-20 Training Code

Training code for the SVDF-20 multilingual singing voice deepfake detection dataset.

## Models

8 state-of-the-art models: AASIST, RawGAT_ST, RawNet2, SpecRNet, Whisper, SSLModel, Conformer, RawNetLite

## Setup

```bash
pip install -r requirements.txt
export SVDF_DATASET_PATH="/path/to/dataset"
```

## Usage

```bash
python train.py --model_name AASIST
```

## Configuration

Edit `config.py` for custom parameters:
- `--model_name`: Model to train
- `--batch_size`: Batch size (default: 32)
- `--num_epochs`: Epochs (default: 25)
- `--lr`: Learning rate (default: 3e-4)

```

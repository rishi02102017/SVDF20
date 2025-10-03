"""
RawNetLite model integration for BhashaBluff baseline
Adapted from the original RawNetLite implementation with pretrained model support
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# Residual block base
class ResBlock(nn.Module):
    """
    A 1D convolutional residual block for processing sequential data.
    """
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + residual)

# RawNetLite model - Original Architecture
class RawNetLite(nn.Module):
    """
    RawNetLite: A lightweight end-to-end architecture for audio deepfake detection.
    Original architecture with GRU for temporal modeling.
    """
    def __init__(self, device, pretrained_path=None):
        super(RawNetLite, self).__init__()
        self.device = device
        
        # Input processing
        self.conv_pre = nn.Conv1d(1, 64, kernel_size=3, stride=1, padding=1)
        self.bn_pre = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()

        # Residual blocks
        self.resblock1 = ResBlock(64)
        self.resblock2 = ResBlock(64)
        self.resblock3 = ResBlock(64)

        # Pooling for GRU
        self.pool = nn.AdaptiveAvgPool1d(64)

        # GRU for temporal modeling
        self.gru = nn.GRU(input_size=64, hidden_size=128, num_layers=1,
                          batch_first=True, bidirectional=True)

        # Classification head
        self.fc1 = nn.Linear(128 * 2, 64)
        self.fc2 = nn.Linear(64, 2)  # 2 classes for CrossEntropyLoss
        
        # Load pretrained weights if provided
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"Loading pretrained RawNetLite from: {pretrained_path}")
            self.load_pretrained(pretrained_path)
        
    def load_pretrained(self, pretrained_path):
        """Load pretrained weights"""
        try:
            checkpoint = torch.load(pretrained_path, map_location=self.device, weights_only=False)
            # Handle different checkpoint formats
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Load weights, ignoring size mismatches
            model_dict = self.state_dict()
            pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)
            print(f"Successfully loaded {len(pretrained_dict)} pretrained parameters")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights: {e}")
            print("Training from scratch...")
        
    def forward(self, x):
        # Reshape input from [B, T] to [B, 1, T] for Conv1d
        nb_samp = x.shape[0]
        len_seq = x.shape[1]
        x = x.view(nb_samp, 1, len_seq)
        
        # x: [B, 1, T]
        x = self.relu(self.bn_pre(self.conv_pre(x)))     # [B, 64, T]
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.pool(x)                                 # [B, 64, 64]

        x = x.transpose(1, 2)                            # [B, 64, 64] → [B, seq, feat]
        output, _ = self.gru(x)                          # [B, 64, 256]
        hidden_output = output[:, -1, :]                 # Last step → [B, 256] (use as hidden)

        x = self.fc1(hidden_output)                      # [B, 64]
        x = self.fc2(x)                                  # [B, 2]
        return x, hidden_output  # Return logits for 2 classes and hidden output (consistent with other models)

def prepare_model(device, pretrained_path=None):
    """Prepare RawNetLite model for training"""
    model = RawNetLite(device, pretrained_path)
    model = model.to(device)
    return model

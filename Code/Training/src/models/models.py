"""
Model factory for SVDF-20 paper models
Supports the 8 models evaluated in the paper:
AASIST, RawGAT_ST, RawNet2, SpecRNet, Whisper, SSLModel, Conformer, RawNetLite
"""
from typing import Dict
import yaml
from src.models import (
    specrnet,
    whisper_main,
    rawnetlite,
    AASIST,
    RawGAT_ST,
    RawNet2,
    Conformer
)


def get_model(model_name: str, device: str):
    """
    Get model instance for the 8 models used in SVDF-20 paper:
    AASIST, RawGAT_ST, RawNet2, SpecRNet, Whisper, SSLModel, Conformer, RawNetLite
    """
    if model_name == "specrnet" or model_name == "SpecRNet":
        # SpecRNet implementation
        with open("./configs/training/specrnet.yaml", "r") as f:
            config = yaml.safe_load(f)
        model_config = config["model"]
        model_name, model_parameters = model_config["name"], model_config["parameters"]
        config = model_parameters
        model = specrnet.FrontendSpecRNet(device=device, **config)
        model = model.to(device)
        return model
    elif model_name == "whisper":
        # Whisper model
        from src.models import whisper_main
        with open("./configs/training/whisper.yaml", "r") as f:
            config = yaml.safe_load(f)
        model_config = config["model"]
        model_name, model_parameters = model_config["name"], model_config["parameters"]
        config = model_parameters
        return whisper_main.WhisperModel(
            input_channels=config.get("input_channels", 1),
            freeze_encoder=config.get("freeze_encoder", False),
            device=device,
        )
    elif model_name == "rawnetlite":
        # RawNetLite model
        from src.models import rawnetlite
        with open("./configs/training/rawnetlite.yaml", "r") as f:
            config = yaml.safe_load(f)
        model_config = config["model"]
        model_name, model_parameters = model_config["name"], model_config["parameters"]
        config = model_parameters
        return rawnetlite.RawNetLite(
            input_channels=config.get("input_channels", 1),
            device=device,
        )
    elif model_name == "rawnet2":
        # RawNet2 model
        from src.models import RawNet2
        with open("./configs/training/rawnet2.yaml", "r") as f:
            config = yaml.safe_load(f)
        model_config = config["model"]
        model_name, model_parameters = model_config["name"], model_config["parameters"]
        config = model_parameters
        return RawNet2.RawNet2(
            input_channels=config.get("input_channels", 1),
            device=device,
        )
    elif model_name == "aasist":
        # AASIST model
        from src.models import AASIST
        with open("./configs/training/aasist.yaml", "r") as f:
            config = yaml.safe_load(f)
        model_config = config["model"]
        model_name, model_parameters = model_config["name"], model_config["parameters"]
        config = model_parameters
        return AASIST.AASIST(
            input_channels=config.get("input_channels", 1),
            device=device,
        )
    elif model_name == "rawgat_st":
        # RawGAT_ST model
        from src.models import RawGAT_ST
        with open("./configs/training/rawgat_st.yaml", "r") as f:
            config = yaml.safe_load(f)
        model_config = config["model"]
        model_name, model_parameters = model_config["name"], model_config["parameters"]
        config = model_parameters
        return RawGAT_ST.RawGAT_ST(
            input_channels=config.get("input_channels", 1),
            device=device,
        )
    elif model_name == "conformer":
        # Conformer model
        from src.models import Conformer
        with open("./configs/training/conformer.yaml", "r") as f:
            config = yaml.safe_load(f)
        model_config = config["model"]
        model_name, model_parameters = model_config["name"], model_config["parameters"]
        config = model_parameters
        return Conformer.Conformer(
            input_channels=config.get("input_channels", 1),
            device=device,
        )
    elif model_name == "sslmodel":
        # SSLModel (Self-supervised learning model)
        # This might be implemented in whisper_main.py or as a separate model
        from src.models import whisper_main
        with open("./configs/training/sslmodel.yaml", "r") as f:
            config = yaml.safe_load(f)
        model_config = config["model"]
        model_name, model_parameters = model_config["name"], model_config["parameters"]
        config = model_parameters
        return whisper_main.SSLModel(
            input_channels=config.get("input_channels", 1),
            device=device,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}. Supported models: specrnet, whisper, rawnetlite, rawnet2, aasist, rawgat_st, conformer, sslmodel")
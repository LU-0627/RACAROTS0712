import torch

from models.carots.modeling_carots import CAROTS
from models.timesnet.modeling_timesnet import TimesNet


def build_model(cfg, device=None):
    model_name = cfg.MODEL.NAME

    model_mapping = {
        "CAROTS": CAROTS,
        "TIMESNET": TimesNet,
    }

    # RDCAROTS models
    if model_name in ["RDCAROTS", "RDCAROTS-no-regime", "RDCAROTS-no-delay-negative", "RDCAROTS-single-prototype"]:
        from models.rd_carots.modeling_rd_carots import RDCAROTS
        model_mapping[model_name] = RDCAROTS

    # Device must be explicitly specified
    if device is None:
        if hasattr(cfg, 'DEVICE'):
            device_str = cfg.DEVICE
        elif hasattr(cfg, 'RDCAROTS') and hasattr(cfg.RDCAROTS, 'DEVICE'):
            device_str = cfg.RDCAROTS.DEVICE
        else:
            raise ValueError(
                "Device must be explicitly specified via cfg.DEVICE or cfg.RDCAROTS.DEVICE. "
                "Do not rely on automatic CUDA detection."
            )
        device = torch.device(device_str)

        # Verify CUDA availability if cuda device requested
        if 'cuda' in device_str and not torch.cuda.is_available():
            raise RuntimeError(
                f"Device '{device_str}' requested but CUDA is not available. "
                "Please install CUDA PyTorch or use CPU device explicitly."
            )

    if model_name in model_mapping:
        if model_name.startswith("RDCAROTS"):
            model = model_mapping[model_name](cfg, device=device)
        else:
            model = model_mapping[model_name](cfg)
            model = model.to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    return model

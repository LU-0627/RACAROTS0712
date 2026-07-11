import torch

from models.carots.modeling_carots import CAROTS
from models.timesnet.modeling_timesnet import TimesNet


def build_model(cfg):
    model_name = cfg.MODEL.NAME

    model_mapping = {
        "CAROTS": CAROTS,
        "TIMESNET": TimesNet,
    }

    # RDCAROTS models
    if model_name in ["RDCAROTS", "RDCAROTS-no-regime", "RDCAROTS-no-delay-negative", "RDCAROTS-single-prototype"]:
        from models.rd_carots.modeling_rd_carots import RDCAROTS
        model_mapping[model_name] = RDCAROTS

    if model_name in model_mapping:
        model = model_mapping[model_name](cfg)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Device handling - use device parameter instead of hardcoded cuda()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    return model

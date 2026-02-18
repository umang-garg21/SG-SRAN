from utils.config_utils import ConfigNamespace
from models.reynolds_qsr import Reynolds_QSR

MODEL_REGISTRY = {
    "reynolds_qsr": Reynolds_QSR,
}


def build_model(cfg):
    model_type = cfg.model_type.lower()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_REGISTRY[model_type](cfg)

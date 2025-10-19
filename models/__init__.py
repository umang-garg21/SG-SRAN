from models.reynolds_qsr import Reynolds_QSR
from models.quaternion_srnet import QuaternionSRNet
from utils.config_utils import ConfigNamespace

MODEL_REGISTRY = {
    "reynolds_qsr": Reynolds_QSR,
    "quaternion_srnet": QuaternionSRNet,
}


def build_model(cfg):
    model_type = cfg.model_type.lower()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_REGISTRY[model_type](cfg)

# models/__init__.py
from models.reynolds_qsr import Reynolds_QSR

# from models.qrbsa_1d import QRBSA_1D

MODEL_REGISTRY = {
    "Reynolds_QSR": Reynolds_QSR,
    # "QRBSA_1D": QRBSA_1D,
}


def make_model(args):
    model_type = args.model.get("type", "Reynolds_QSR")
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_REGISTRY[model_type](args)

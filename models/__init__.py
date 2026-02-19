from utils.config_utils import ConfigNamespace
from models.reynolds_qsr import Reynolds_QSR
from models.autoencoder import FCCAutoEncoder
from models.autoencoder_learnable import FCCLearnableDecoderAutoEncoder
from models.invariant_sr import InvariantSRModel


def _build_fcc_autoencoder(cfg):
    return FCCAutoEncoder(
        device=getattr(cfg, "device", None),
        grid_res=getattr(cfg, "grid_res", 100_000),
    )


def _build_fcc_autoencoder_learnable_decoder(cfg):
    return FCCLearnableDecoderAutoEncoder(
        device=getattr(cfg, "device", None),
        hidden_dim=getattr(cfg, "decoder_hidden_dim", 128),
        num_layers=getattr(cfg, "decoder_num_layers", 3),
        dropout=getattr(cfg, "decoder_dropout", 0.0),
    )


def _build_invariant_sr(cfg):
    return InvariantSRModel(
        device=getattr(cfg, "device", None),
        upsample_factor=getattr(cfg, "scale", 4),
        decoder_grid_res=getattr(cfg, "grid_res", 10_000),
        kernel_size=getattr(cfg, "kernel_size", 3),
    )

MODEL_REGISTRY = {
    "reynolds_qsr": Reynolds_QSR,
    "fcc_autoencoder": _build_fcc_autoencoder,
    "fcc_autoencoder_learnable_decoder": _build_fcc_autoencoder_learnable_decoder,
    "invariant_sr": _build_invariant_sr,
}


def build_model(cfg):
    model_type = cfg.model_type.lower()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_REGISTRY[model_type](cfg)

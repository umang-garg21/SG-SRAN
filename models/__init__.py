from utils.config_utils import ConfigNamespace
from models.reynolds_qsr import Reynolds_QSR
from models.autoencoder import FCCAutoEncoder
from models.autoencoder_learnable import FCCLearnableDecoderAutoEncoder
from models.invariant_sr import InvariantSRModel


def _build_fcc_autoencoder(cfg):
    decoder_config = {
        "decoder_cubochoric_resolution": getattr(cfg, "decoder_cubochoric_resolution", 3),
        "decoder_lookup_resolution": getattr(
            cfg,
            "decoder_lookup_resolution",
            getattr(cfg, "decoder_cubochoric_resolution", 3),
        ),
        "decoder_lookup_chunk_size": getattr(cfg, "decoder_lookup_chunk_size", 8192),
        "decoder_lookup_npy_path": getattr(cfg, "decoder_lookup_npy_path", None),
        "decoder_lookup_rebuild": getattr(cfg, "decoder_lookup_rebuild", False),
        "decoder_lookup_refine_steps": getattr(cfg, "decoder_lookup_refine_steps", 0),
        "decoder_lookup_refine_lr": getattr(cfg, "decoder_lookup_refine_lr", 0.05),
        "decoder_learnable_hidden_dim": getattr(cfg, "decoder_learnable_hidden_dim", 256),
        "decoder_learnable_num_layers": getattr(cfg, "decoder_learnable_num_layers", 3),
        "decoder_learnable_dropout": getattr(cfg, "decoder_learnable_dropout", 0.0),
        "decoder_learnable_ckpt_path": getattr(cfg, "decoder_learnable_ckpt_path", None),
        "decoder_learnable_ckpt_strict": getattr(cfg, "decoder_learnable_ckpt_strict", True),
        "decoder_num_starts": getattr(cfg, "decoder_num_starts", 6),
        "decoder_steps": getattr(cfg, "decoder_steps", 25),
        "decoder_lr": getattr(cfg, "decoder_lr", 0.08),
        "decoder_w6": getattr(cfg, "decoder_w6", 0.5),
        "decoder_log_optimization": getattr(cfg, "decoder_log_optimization", False),
        "decoder_log_every": getattr(cfg, "decoder_log_every", 1),
    }

    return FCCAutoEncoder(
        device=getattr(cfg, "device", None),
        grid_res=getattr(cfg, "grid_res", 100_000),
        decoder_backend=getattr(cfg, "decoder_backend", "optimizing"),
        decoder_config=decoder_config,
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
        decoder_backend=getattr(cfg, "decoder_backend", "optimizing"),
        decoder_cubochoric_resolution=getattr(cfg, "decoder_cubochoric_resolution", 3),
        decoder_num_starts=getattr(cfg, "decoder_num_starts", 6),
        decoder_steps=getattr(cfg, "decoder_steps", 25),
        decoder_lr=getattr(cfg, "decoder_lr", 0.08),
        decoder_w6=getattr(cfg, "decoder_w6", 0.5),
        kernel_size=getattr(cfg, "kernel_size", 3),
        learned_decoder_hidden_dim=getattr(cfg, "learned_decoder_hidden_dim", 64),
        train_decode_mode=getattr(cfg, "train_decode_mode", "learnable"),
        eval_decode_mode=getattr(cfg, "eval_decode_mode", "spherical"),
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

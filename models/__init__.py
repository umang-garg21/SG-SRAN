"""Core model exports for the IsoEmbeddingSRAttn workflow."""

from models.bicubic_f_interpolate_sr import QuaternionBicubicFInterpolateSR

__all__ = [
    "QuaternionBicubicFInterpolateSR",
]

try:
    from models.SR_double_conv_SRattn import (
        AttentionBlock,
        CubochoricOptimizingLocalIsoDecoder,
        EquivariantSpatialConv,
        EquivariantTransposeConv,
        IsoEmbeddingSRAttn,
        LearnableA1QuaternionDecoder,
        LocalIsoCrystalEncoder,
    )
    from models.local_iso_embedding import (
        LocalIsoCrystalEmbedding,
        build_fcc_syms_mtex,
        build_hcp_syms_mtex,
        build_local_iso_fcc_embedding,
        build_local_iso_hcp_embedding,
        cubic_group_O,
        dihedral_group_D6_paper,
        dihedral_group_D6_zaxis,
    )
except ModuleNotFoundError as exc:
    if exc.name not in {"e3nn", "orix"}:
        raise
else:
    __all__.extend(
        [
            "AttentionBlock",
            "CubochoricOptimizingLocalIsoDecoder",
            "EquivariantSpatialConv",
            "EquivariantTransposeConv",
            "IsoEmbeddingSRAttn",
            "LearnableA1QuaternionDecoder",
            "LocalIsoCrystalEncoder",
            "LocalIsoCrystalEmbedding",
            "build_fcc_syms_mtex",
            "build_hcp_syms_mtex",
            "build_local_iso_fcc_embedding",
            "build_local_iso_hcp_embedding",
            "cubic_group_O",
            "dihedral_group_D6_paper",
            "dihedral_group_D6_zaxis",
        ]
    )

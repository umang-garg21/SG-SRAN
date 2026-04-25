"""Core model exports for the IsoEmbeddingSRAttn workflow."""

from models.bicubic_f_interpolate_sr import QuaternionBicubicFInterpolateSR

__all__ = [
    "QuaternionBicubicFInterpolateSR",
]

try:
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
    from models.SR_ocrp import (
        ClusterSlotBuilder,
        CosineMaskedEquivariantSpatialConv,
        IsoEmbeddingSROCRP,
        MedoidSlotContextBuilder,
        OCRPPatchUpsampler,
        PatchSlotRouter,
        QuaternionBankClusterer,
        SharedTPPatchProposalHead,
        WithinSlotInvariantPool,
    )
    from models.SR_4x1_ocrp import (
        IsoEmbedding4x1SROCRP,
        OCRP4x1PatchUpsampler,
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
            "QuaternionBankClusterer",
            "ClusterSlotBuilder",
            "MedoidSlotContextBuilder",
            "CosineMaskedEquivariantSpatialConv",
            "WithinSlotInvariantPool",
            "PatchSlotRouter",
            "SharedTPPatchProposalHead",
            "OCRPPatchUpsampler",
            "IsoEmbeddingSROCRP",
            "OCRP4x1PatchUpsampler",
            "IsoEmbedding4x1SROCRP",
        ]
    )

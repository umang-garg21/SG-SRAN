"""
Dedicated OCRP training entrypoint.

This reuses the shared quaternion-SR trainer and relies on the experiment
config to select `models.SR_ocrp.IsoEmbeddingSROCRP`.
"""

from training.train_iso_embedding_sr_attn import main


if __name__ == "__main__":
    main()

"""The supervoxel autoencoder's own layers.

What the framework in `train/vendor/trellis2/` provides is the sparse tensor type and the blocks
built on it. What is here is the architecture assembled from those, in three files:

    encoder_cvt.py   the CVT-conditioned sparse U-Net encoder
    layers.py        positional encoding, KNN self-attention, bidirectional cross-attention
    quantizer.py     the residual FSQ codebook the supervoxel tokens are drawn from

`vae.py`, one level up, composes these into the encoder and decoder the released checkpoints load
into. Nothing here is an entry point.
"""

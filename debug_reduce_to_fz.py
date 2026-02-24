import numpy as np
from orix.quaternion import Orientation, symmetry

def reduce_to_fz_with_ops_orix(
    q_xyzw: np.ndarray,              # scalar-last input (...,4)
    sym=symmetry.O,                  # use symmetry.O for proper cubic (24); symmetry.Oh for Laue (48)
    verbose: bool = True,
    return_order: str = "xyzw",      # "xyzw" or "wxyz"
):
    """
    Reduce quaternions into the symmetry-reduced zone (fundamental zone) and
    return the symmetry operators used.

    Returns
    -------
    q_reduced : ndarray (...,4)
        Reduced quaternions.
    s_l : orix.quaternion.Rotation (or Symmetry-like)
        Left operators used per quaternion (vectorized container).
    s_r : orix.quaternion.Rotation (or Symmetry-like)
        Right operators used per quaternion.
    """
    q_xyzw = np.asarray(q_xyzw, dtype=np.float64)
    assert q_xyzw.shape[-1] == 4, f"Expected (...,4), got {q_xyzw.shape}"

    orig_shape = q_xyzw.shape[:-1]

    # xyzw -> wxyz for orix
    q_wxyz = np.stack([q_xyzw[..., 3], q_xyzw[..., 0], q_xyzw[..., 1], q_xyzw[..., 2]], axis=-1)

    # normalize + hemisphere
    n = np.linalg.norm(q_wxyz, axis=-1, keepdims=True)
    n = np.where(n == 0, 1.0, n)
    q_wxyz = q_wxyz / n
    q_wxyz = np.where(q_wxyz[..., :1] < 0, -q_wxyz, q_wxyz)

    # Build Orientation and map into reduced zone (FZ) with ops
    ori = Orientation(q_wxyz.reshape(-1, 4), symmetry=sym)

    # This is the call you referenced:
    q_red, s_l, s_r = ori.map_into_symmetry_reduced_zone_with_ops(verbose=verbose)

    # q_red is an Orientation; get back quaternions
    q_red_wxyz = q_red.data.reshape(orig_shape + (4,))

    if return_order.lower() == "wxyz":
        return q_red_wxyz, s_l, s_r
    elif return_order.lower() == "xyzw":
        q_red_xyzw = np.stack([q_red_wxyz[..., 1], q_red_wxyz[..., 2], q_red_wxyz[..., 3], q_red_wxyz[..., 0]], axis=-1)
        return q_red_xyzw, s_l, s_r
    else:
        raise ValueError("return_order must be 'xyzw' or 'wxyz'")
    
if __name__ == "__main__":

    q_xyzw = np.load("/data/warren/materials/EBSD/IN718_2D_SR_x4/Test/Original_Data/Open_718_Test_hr_x_block_0.npy")  # (H,W,4) scalar-last

    q_red_xyzw, s_l, s_r = reduce_to_fz_with_ops_orix(
        q_xyzw,
        sym=symmetry.Oh,       # recommended for cubic proper
        verbose=True,
        return_order="xyzw"
    )

    H, W, _ = q_xyzw.shape

    # s_l.data and s_r.data are (N,4) where N=H*W in wxyz order
    sL_HW = s_l.reshape(H, W)
    sR_HW = s_r.reshape(H, W)
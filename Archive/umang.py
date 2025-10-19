import inspect, orix
from orix.quaternion import Orientation

print(inspect.getfile(Orientation))
import numpy as np
from orix.quaternion import Orientation
from orix.quaternion.symmetry import get_point_group
from orix.vector import Vector3d
from orix.plot import IPFColorKeyTSL


def quaternion_to_ipf(quats, axis="Z"):
    ori = Orientation(np.asarray(quats), symmetry=SYM.Oh)

    key = IPFColorKeyTSL(SYM.Oh, direction=getattr(Vector3d, f"{axis.lower()}vector")())
    return key.orientation2color(~ori)


# plot rgbs_reshaped as an image
import matplotlib.pyplot as plt

plt.figure(figsize=(6, 6))
plt.imshow(quaternion_to_ipf(a, axis="Y"))
plt.axis("off")
plt.title("RGB Image from Quaternions")
plt.show()

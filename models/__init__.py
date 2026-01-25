from models.reynolds_qsr import Reynolds_QSR
from models.quaternion_srnet import QuaternionSRNet
from models.quaternion_res_srnet import Quaternion_res_SRNet  # Fixed: import correct model
from models.quat_pool import QuaternionPoolSRNet
from models.quat_k import ProgressiveQuaternionSRNet
from utils.config_utils import ConfigNamespace
from models.quaternion_deepconv_srnet import Quaternion_DeepConv_SRNet
from models.reynolds_res_qsrnet import Reynolds_res_QSRNet
from models.reynolds_deepconv_qsrnet import Reynolds_DeepConv_QSRNet
from models.reynolds_qrbsa import Equivariant_QRBSA
from models.reynolds_qrbsa_different_upsampler import Reynolds_QRBSA_Different_Upsampler
from models.network_slerp import Network_Slerp
from models.network_slerp_lifted_Ops import Network_Slerp_Lifted_Ops

MODEL_REGISTRY = {
    "reynolds_qsr": Reynolds_QSR,
    "quaternion_srnet": QuaternionSRNet,
    "quaternion_res_srnet": Quaternion_res_SRNet,  # Fixed: use correct model class
    "quaternion_pool_srnet": QuaternionPoolSRNet,
    "quaternion_kernel": ProgressiveQuaternionSRNet,
    "quaternion_deepconv_srnet": Quaternion_DeepConv_SRNet,
    "reynolds_res_qsrnet": Reynolds_res_QSRNet,  # Assuming Reynolds_QSR is the correct class for this type
    "reynolds_deepconv_qsrnet": Reynolds_DeepConv_QSRNet,
    "equivariant_qrbsa": Equivariant_QRBSA,
    "reynolds_qrbsa_different_upsampler": Reynolds_QRBSA_Different_Upsampler,
    "network_slerp": Network_Slerp,
    "network_slerp_lifted_ops": Network_Slerp_Lifted_Ops,
}


def build_model(cfg):
    model_type = cfg.model_type.lower()
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_REGISTRY[model_type](cfg)

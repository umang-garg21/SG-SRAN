import matplotlib.pyplot as plt
import numpy as np

from orix.quaternion import Orientation, symmetry
from orix.sampling import get_sample_fundamental
from orix.vector import Vector3d


plt.rcParams.update(
    {
        "axes.grid": True,
        "figure.figsize": (15, 5),
        "font.size": 20,
        "lines.linewidth": 2,
    }
)

pg432 = symmetry.Oh

resolution2 = 1

rot_cube = get_sample_fundamental(
    resolution2, point_group=pg432, method="cubochoric"
)

# rot_cube = get_sample_fundamental(point_group=pg432, method="cubochoric", semi_edge_steps=67)  # Gives identical results
rot_cube

ori_cube = Orientation(rot_cube, symmetry=pg432)
ori_cube


rot_euler = get_sample_fundamental(
    resolution2, point_group=pg432, method="haar_euler"
)
ori_euler = Orientation(rot_euler, symmetry=pg432)
print(ori_euler.size)



rot_quat = get_sample_fundamental(
    resolution2, point_group=pg432, method="quaternion"
)


ori_quat = Orientation(rot_quat, symmetry=pg432)
print(ori_quat.size)

np.random.seed(42)  # For reproducibility of the random sample
n = 10000
ori_cube2 = ori_cube.get_random_sample(n)
ori_euler2 = ori_euler.get_random_sample(n)
ori_quat2 = ori_quat.get_random_sample(n)

fig = plt.figure(figsize=(15, 10))
scatter_kwargs = dict(
    projection="rodrigues",
    figure=fig,
    wireframe_kwargs=dict(color="k", linewidth=1, alpha=0.1),
    s=5,
)

ori_cube2.scatter(position=231, c="C0", **scatter_kwargs)
ori_euler2.scatter(position=232, c="C1", **scatter_kwargs)
ori_quat2.scatter(position=233, c="C2", **scatter_kwargs)

ori_cube2.scatter(position=234, c="C0", **scatter_kwargs)
ori_euler2.scatter(position=235, c="C1", **scatter_kwargs)
ori_quat2.scatter(position=236, c="C2", **scatter_kwargs)

titles = ["cubochoric", "haar_euler", "quaternion"]
for i, title in zip([0, 1, 2], titles):
    fig.axes[i].view_init(elev=90, azim=0)
    fig.axes[i].set_title(titles[i])
for i in [3, 4, 5]:
    fig.axes[i].view_init(elev=0, azim=0)

    
directions = Vector3d(((1, 0, 0), (0, 1, 0), (0, 0, 1)))

ori_cube2.scatter("ipf", direction=directions, c="C0", s=5)


ori_euler2.scatter("ipf", direction=directions, c="C1", s=5)

ori_quat2.scatter("ipf", direction=directions, c="C2", s=5)

# from pathlib import Path
# import torch

# from training.config_utils import load_and_prepare_config
# from training.data_loading import build_dataloader
# from models.autoencoder import FCCAutoEncoder
# from utils.symmetry_utils import resolve_symmetry
# from visualization.visualize_sr_results import render_input_output_side_by_side

# exp_dir = Path('experiments/IN718/debug_x4')
# config_path = exp_dir / 'config_smoke.json'
# run_cfg_path = exp_dir / 'logs' / 'run_config_eval_only_lookup.json'
# cfg = load_and_prepare_config(config_path, run_cfg_path)

# # Force eval-only / no-learning fast lookup decode path
# cfg.model.type = 'fcc_autoencoder'
# cfg.decoder_backend = 'lookup'
# cfg.decoder_lookup_resolution = 1
# cfg.decoder_w6 = float(getattr(cfg, 'decoder_w6', 0.5))

# # Build test loader (use existing smoke behavior if set)
# loader = build_dataloader(
#     dataset_root=cfg.dataset_root,
#     split='Test',
#     batch_size=cfg.batch_size,
#     num_workers=cfg.num_workers,
#     preload=cfg.preload,
#     preload_torch=cfg.preload_torch,
#     pin_memory=cfg.pin_memory,
#     take_first=10 if cfg.smoke_test else None,
#     seed=int(getattr(cfg, 'seed', 42)),
# )

# model = FCCAutoEncoder(
#     device='cuda' if torch.cuda.is_available() else 'cpu',
#     grid_res=int(getattr(cfg, 'grid_res', 1000)),
#     decoder_backend='lookup',
#     decoder_lookup_resolution=int(getattr(cfg, 'decoder_lookup_resolution', 1)),
#     decoder_w6=float(getattr(cfg, 'decoder_w6', 0.5)),
# )
# model.eval()
# device = model.device
# sym_class = resolve_symmetry(getattr(cfg, 'symmetry_group', 'Oh'))

# out_dir = exp_dir / 'visualizations' / 'eval_only_test_lookup'
# out_dir.mkdir(parents=True, exist_ok=True)

# saved = 0
# with torch.no_grad():
#     for _, hr in loader:
#         bsz = int(hr.shape[0])
#         for b in range(bsz):
#             q = hr[b].permute(1, 2, 0).reshape(-1, 4).to(device)
#             q = q / q.norm(dim=1, keepdim=True).clamp_min(1e-12)
#             q_out = model(q, normalize_input=True)

#             h, w = int(hr[b].shape[1]), int(hr[b].shape[2])
#             q_in_np = q.reshape(h, w, 4).detach().cpu().numpy()
#             q_out_np = q_out.reshape(h, w, 4).detach().cpu().numpy()

#             out_png = out_dir / f'input_output_ipf_res_1_{saved:03d}.png'
#             render_input_output_side_by_side(
#                 input_q_arr=q_in_np,
#                 output_q_arr=q_out_np,
#                 sym_class=sym_class,
#                 out_png=str(out_png),
#                 ref_dir='ALL',
#                 include_key=True,
#                 overwrite=True,
#                 format_input=False,
#                 dpi=250,
#             )
#             saved += 1

# print('saved_count', saved)
# print('out_dir', out_dir)
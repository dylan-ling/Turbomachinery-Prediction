# export_vtu.py

import os
import numpy as np
import torch
import trimesh
from torch_geometric.nn.pool import knn_graph

from newdata import Rotor37Dataset
from edgeconvmodel import DGCNNReg
from visualize import save_multiple_field

DATA_ROOT   = "/vols/numeca_nfs01/dling/rotor"
SPLIT       = "test"
NUM_POINTS  = 2048
K           = 20
MODEL_PATH  = os.path.join(DATA_ROOT, "best_rel_model.pth")
OUT_DIR     = os.path.join(DATA_ROOT, "pred_vtu")
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

stats      = np.load(os.path.join(DATA_ROOT, "field_bc_stats.npz"))
field_mean = torch.tensor(stats["field_mean"], device=DEVICE)
field_std  = torch.tensor(stats["field_std"],  device=DEVICE)

ds  = Rotor37Dataset(DATA_ROOT, split=SPLIT, num_points=NUM_POINTS)
bc_names  = ds.bc_names
bc_dicts  = ds.bc_dicts
bc_mean   = ds.bc_mean
bc_std    = ds.bc_std
blades    = ds.blades

# Load trained model
input_dim = ds[0].x.shape[1]
model     = DGCNNReg(k=K, output_dim=4).to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Make directory for VTUs 
os.makedirs(OUT_DIR, exist_ok=True)
print(f"Writing VTUs into {OUT_DIR}")

# Loop over blades & export VTU 
for blade in blades:
    print(f"  • {blade} …", end="", flush=True)

    # Load mesh geometry
    mesh   = trimesh.load(os.path.join(DATA_ROOT, "stl", blade + ".stl"),
                          process=False)
    points = mesh.vertices       # (V,3)
    faces  = mesh.faces          # (F,3)

    # Load fields + coords
    arr        = np.loadtxt(os.path.join(DATA_ROOT, "csv", blade + ".csv"),
                            delimiter=",", skiprows=1)
    gt_fields  = arr[:, :4]      # (N,4)
    raw_coords = arr[:, 4:7]

    _, _, tri_ids = mesh.nearest.on_surface(raw_coords)

    # Form per-vertex features & run inference
    normals = mesh.face_normals[tri_ids]

    coords = raw_coords - raw_coords.mean(axis=0)
    coords = coords / np.linalg.norm(coords, axis=1).max()

    bid    = int(blade.split("_")[-1])
    raw_bc = np.array([bc_dicts[n][bid] for n in bc_names],
                      dtype=np.float32)
    bc_vec = (raw_bc - bc_mean) / bc_std

    feat      = np.hstack([coords, normals,
                           np.tile(bc_vec[None, :], (coords.shape[0], 1))])
    x_in      = torch.from_numpy(feat).float().to(DEVICE)
    batch_i   = torch.zeros(x_in.size(0), dtype=torch.long,
                            device=DEVICE)
    edge_idx  = knn_graph(x_in, k=K, batch=batch_i, loop=False)

    with torch.no_grad():
        x1    = model.conv1(x_in,  edge_idx)
        x2    = model.conv2(x1,     edge_idx)
        x3    = model.conv3(x2,     edge_idx)
        x4    = model.conv4(x3,     edge_idx)
        x5    = model.conv5(x4,     edge_idx)
        x_cat = torch.cat([x1, x2, x3, x4, x5], dim=1)
        out_n = model.mlp(x_cat)

    pred_n    = out_n.cpu().numpy()
    pred_phys = pred_n * field_std.cpu().numpy() + field_mean.cpu().numpy()
    abs_err   = np.abs(pred_phys - gt_fields)

    # Assemble point_data for meshio
    names = ["Density", "Entropy", "Pressure", "Temp"]
    field_dict = {}
    for i, name in enumerate(names):
        field_dict[f"{name}_gt"]   = gt_fields[:, i]
        field_dict[f"{name}_pred"] = pred_phys[:,   i]
        field_dict[f"{name}_err"]  = abs_err[:,     i]

    # Write VTU
    out_vtu = os.path.join(OUT_DIR, f"{blade}.vtu")
    save_multiple_field(out_vtu, points, faces, field_dict)
    print(" done.")

print("All VTUs written.")

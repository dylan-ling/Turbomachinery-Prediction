# data.py

import os
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
# from torch_geometric.data import InMemoryDataset
import trimesh

# class Rotor37Dataset(InMemoryDataset):
class Rotor37Dataset(Dataset):

    def __init__(self, root, split='train', num_points=1024):
        """
        Reads each Blade CSV + STL, normalizes flow fields & BCs, and returns per-point
        features [x,y,z,nx,ny,nz, bc1…bc7] in Data.x
        """
        self.root       = root
        self.num_points = num_points
        # self.split = split

        # 1) Load field_bc_stats.npz
        stats = np.load(os.path.join(root, 'field_bc_stats.npz'))
        self.field_mean = stats['field_mean']   # (4,)
        self.field_std  = stats['field_std']    # (4,)
        self.bc_mean    = stats['bc_mean']      # (7,)
        self.bc_std     = stats['bc_std']       # (7,)
        valid_ids       = stats['blade_ids'].tolist()

        # 2) Load BC dicts
        self.bc_names = [
            "Static_pressure",
            "Static_pressure_ratio",
            "Static_temperature_ratio",
            "Isentropic_efficiency",
            "Polytropic_efficiency",
            "Absolute_total_pressure_ratio",
            "Absolute_total_temperature_ratio",
        ]
        self.bc_dicts = {
            name: np.load(os.path.join(root, 'Scalars_input', 'Dict', name + '.npy'),
                          allow_pickle=True).item()
            for name in self.bc_names
        }

        # 3) Read split list
        split_file = os.path.join(root, f'{split}_files.txt')
        #split_file = os.path.join(root, f'{self.split}_files.txt')

        with open(split_file) as f:
            blades = [line.strip() for line in f if line.strip()]

        self.blades = [b for b in blades
                       if int(b.split('_')[-1]) in valid_ids]
    
        # super().__init__(root)
        # data_path = os.path.join(root, f'{self.split}_data.pt')
        # self.data, self.slices = torch.load(data_path)

    # @property
    # def processed_file_names(self):
        # return [f'{self.split}_data.pt']
    
    def __len__(self):
        return len(self.blades)
    def __getitem__(self, idx):
        blade = self.blades[idx]
        bid   = int(blade.split('_')[-1])

        mesh         = trimesh.load(os.path.join(self.root, 'stl', blade + '.stl'),
                                    process=False)

        verts_full = np.asarray(mesh.vertices, dtype=np.float32)
        normals_full = np.asarray(mesh.vertex_normals, dtype=np.float32)  # [V_mesh,3]

        arr    = np.loadtxt(os.path.join(self.root, 'csv', blade + '.csv'),
                            delimiter=',', skiprows=1).astype(np.float32)
        fields = arr[:, :4]    # [V_csv,4]
        coords = arr[:, 4:7]   # [V_csv,3]

        #Normalize flow‐fields per‐point
        fields = (fields - self.field_mean) / self.field_std  # [V,4]

        #Center & unit‐normalize coords
        coords = coords - coords.mean(axis=0)
        coords = coords / np.linalg.norm(coords, axis=1).max()  # [V,3]

        # Build & normalize BC vector
        raw_bc = np.array([ self.bc_dicts[n][bid] for n in self.bc_names ],
                         dtype=np.float32)
        bc_vec = (raw_bc - self.bc_mean) / self.bc_std

        V     = coords.shape[0]
        repl  = V < self.num_points
        idxs  = np.random.choice(V, self.num_points, replace=repl)

        samp_pos    = coords[idxs]     # [P,3]
        samp_fields = fields[idxs]     # [P,4]
        samp_norm   = normals_full[idxs]  # [P,3]

        samp_bc = np.tile(bc_vec[None,:], (self.num_points, 1))

        # Build final per-point feature: coords|normals|bc → dim=13
        x = np.concatenate([samp_pos, samp_norm, samp_bc], axis=1)  # [P,13]

        return Data(
            x   = torch.from_numpy(x).float(),        # [P,13]
            pos = torch.from_numpy(samp_pos).float(), # [P,3] for knn_graph
            y   = torch.from_numpy(samp_fields).float()  # [P,4]


        # data, slices = self.collate(data_list)
        # os.makedir(slelf.processed_dir, exist_ok=True)
        # torch.save((data, slices), oos.path.join(self.processed_dir, f'{self.split}_files.txt')
        )

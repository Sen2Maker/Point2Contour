import os, sys, pickle
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset

class NewDateset(Dataset):

    def __init__(self, p, data_root=None):
        super().__init__()

        if data_root is None:
            raise ValueError("data_root is required.")
        dataset_path = Path(data_root).expanduser().resolve()
        if not dataset_path.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_path}")
        self.dataset_path = dataset_path.as_posix()

        self.mode = p['mode']

        if self.mode in ['train', 'val', 'test']:
            self.names = np.loadtxt(
                os.path.join(self.dataset_path, f'{self.mode}.txt'),
                dtype=str
            )
            self.names = np.atleast_1d(self.names).astype(str)

            if self.mode == 'val':

                print(f"Loaded {len(self.names)} validation samples.")

        else:
            raise NotImplementedError(f"Unsupported dataset mode: {self.mode}")

    def __len__(self):
        return len(self.names)

    def _sample_pkl_path(self, name: str):
        name = str(name)
        base_dir = os.path.join(self.dataset_path, name)
        path = os.path.join(base_dir, 'pc_with_edge_c_full.pkl')

        if not os.path.isfile(path):
            raise FileNotFoundError(f'No pc_with_edge_c_full.pkl found for {name} under {base_dir}')

        return path

    def _load(self, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def collate_fn(self, batch):
        if len(batch) != 1:
            raise ValueError('PreDateset expects batch_size == 1.')
        return batch[0]

    def corners_to_edge_indices_force_include(self, edge_idx: np.ndarray, corner_idx: np.ndarray):
        edge_idx = np.asarray(edge_idx, dtype=np.int64).ravel()
        corner_idx = np.asarray(corner_idx, dtype=np.int64).ravel()

        pos_map = {}
        for i, p in enumerate(edge_idx):
            if p not in pos_map:
                pos_map[p] = i

        corner_pos = np.empty(corner_idx.shape[0], dtype=np.int64)
        edge_list = edge_idx.tolist()
        next_pos = len(edge_list)

        for k, c in enumerate(corner_idx):
            c = int(c)
            pos = pos_map.get(c, -1)
            if pos == -1:
                edge_list.append(c)
                pos_map[c] = next_pos
                corner_pos[k] = next_pos
                next_pos += 1
            else:
                corner_pos[k] = pos

        new_edge_idx = np.asarray(edge_list, dtype=np.int64)
        return corner_pos, new_edge_idx

    def _compute_normals_pca(self, points, knn_idx):
        neighbors = points[knn_idx]  
        means = np.mean(neighbors, axis=1, keepdims=True)
        centered = neighbors - means
        cov = np.matmul(centered.transpose(0, 2, 1), centered)  

        eigvals, eigvecs = np.linalg.eigh(cov)
        normals = eigvecs[:, :, 0]

        view_dirs = -points
        dots = np.sum(normals * view_dirs, axis=1)
        normals[dots < 0] *= -1

        return normals.astype(np.float32)

    def __getitem__(self, idx: int):

        name = str(self.names[idx])
        path = self._sample_pkl_path(name)
        d = self._load(path)

        pc = d['pc']
        N = pc.shape[0]
        pc_KNN_idx = d['pc_KNN_idx']
        block_idx_np = d['block_idx']
        centers_idx = d['centers_idx']

        fp_idx = d['fp_idx']
        fp_dist = d['fp_dist']

        edge_soft_label = d['edge_soft_label']
        corner_gt_xyz_real = d['corner_vertices_xyz']
        base_rays = d['spherical_basedir']

        corner_adj_indptr = d['corner_adj_indptr']
        corner_adj_indices = d['corner_adj_indices']

        n_center = d['n_center']
        n_factor = d['n_factor']

        model_input = {
            'pc': torch.from_numpy(pc).float(),
            'pc_KNN_idx': torch.from_numpy(pc_KNN_idx.astype(np.int64)),
            'centers_idx': torch.from_numpy(centers_idx.astype(np.int64)),
            'block_idx': torch.from_numpy(block_idx_np.astype(np.int64)),
            'fp_idx': torch.from_numpy(fp_idx.astype(np.int64)),
            'fp_dist': torch.from_numpy(fp_dist.astype(np.float32)),

            'edge_soft_label': torch.from_numpy(edge_soft_label.astype(np.float32)),
            'flat_rays': torch.from_numpy(base_rays).float(),

            'corner_xyz': torch.from_numpy(corner_gt_xyz_real).float(),
            'corner_adj_indptr': torch.from_numpy(corner_adj_indptr.astype(np.int64)),
            'corner_adj_indices': torch.from_numpy(corner_adj_indices.astype(np.int64)),

            'n_center': torch.tensor(n_center, dtype=torch.float64),
            'n_factor': torch.tensor(n_factor, dtype=torch.float64),
            'corner_link': torch.from_numpy(d['corner_link']).long(),
        }

        gt = {
            'edge_soft_label': torch.from_numpy(edge_soft_label.astype(np.float32)),
            'n_center': torch.tensor(n_center, dtype=torch.float64),
            'n_factor': torch.tensor(n_factor, dtype=torch.float64),
            'base_rays': torch.from_numpy(base_rays).float(),
            'corner_link': torch.from_numpy(d['corner_link']).long(),
            'corner_xyz': torch.from_numpy(corner_gt_xyz_real).float(),
            'corner_adj_indptr': torch.from_numpy(corner_adj_indptr.astype(np.int64)),
            'corner_adj_indices': torch.from_numpy(corner_adj_indices.astype(np.int64)),
        }

        info = {
            'name': name,
            'path': path,
            'N': int(N),
            'B': int(block_idx_np.shape[0]),
            'K': int(block_idx_np.shape[1]),
            'n_center': n_center,
            'n_factor': n_factor,
        }

        return model_input, gt, info


import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset


class RefinementCacheDataset(Dataset):

    def __init__(self, p, data_root=None):
        super().__init__()

        self.cache_dir = p['cache_dir']

        valid_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.pkl')]
        self.names = [f.replace('.pkl', '') for f in valid_files]
        self.names.sort()

        if len(self.names) == 0:
            raise ValueError(f"No PKL files found in {self.cache_dir}.")

        print(f"Loaded {len(self.names)} cached scenes.")

    def __len__(self):
        return len(self.names)

    def _load(self, path: str):
        with open(path, 'rb') as f:
            return pickle.load(f)

    def collate_fn(self, batch):

        if len(batch) != 1:
            raise ValueError('RefinementCacheDataset requires batch_size=1.')
        return batch[0]

    def __getitem__(self, idx: int):
        name = str(self.names[idx])
        path = os.path.join(self.cache_dir, f"{name}.pkl")

        d = self._load(path)

        model_input = {}
        for k, v in d.items():
            if not k.startswith('gt_'):
                model_input[k] = v

        gt = {}
        if 'gt_corner_xyz' in d:
            gt['corner_xyz'] = d['gt_corner_xyz']
        if 'gt_corner_link' in d:
            gt['corner_link'] = d['gt_corner_link']

        if 'n_center' in d:
            gt['n_center'] = d['n_center']
            gt['n_factor'] = d['n_factor']

        if 'flat_rays' in d:
            gt['base_rays'] = d['flat_rays']

        info = {
            'name': name,
            'path': path,
        }

        return model_input, gt, info

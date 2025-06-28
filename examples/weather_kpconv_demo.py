import numpy as np
import torch
from sklearn.neighbors import KDTree

from datasets.WeatherDataset import WeatherDataset, estimate_kernel_step
from models.blocks import KPConv


def build_common_grid(all_points):
    step = estimate_kernel_step(all_points)
    min_b = all_points.min(axis=0)
    max_b = all_points.max(axis=0)
    xs = np.arange(min_b[0], max_b[0] + step, step)
    ys = np.arange(min_b[1], max_b[1] + step, step)
    zs = np.arange(min_b[2], max_b[2] + step, step)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3).astype(np.float32)
    return grid, float(step)


def run_batch(batch):
    # collect all points from every variable
    all_points = []
    variables = list(batch[0].keys())
    for sample in batch:
        for var in variables:
            all_points.append(sample[var]["points"])
    all_points = np.concatenate(all_points, axis=0)

    grid, step = build_common_grid(all_points)
    radius = step * 2.5

    kpconv = KPConv(
        kernel_size=15,
        p_dim=3,
        in_channels=1,
        out_channels=8,
        KP_extent=step * 1.2,
        radius=radius,
    )

    q_pts = torch.from_numpy(grid)
    results = {}
    for var in variables:
        pts = np.concatenate([s[var]["points"] for s in batch], axis=0)
        vals = np.concatenate([s[var]["values"] for s in batch], axis=0)[:, None]
        tree = KDTree(pts)
        neigh = tree.query_radius(grid, r=radius)
        max_len = max(len(n) for n in neigh)
        neigh_inds = np.full((len(grid), max_len), pts.shape[0], dtype=np.int64)
        for i, inds in enumerate(neigh):
            neigh_inds[i, : len(inds)] = inds
        s_pts = torch.from_numpy(pts)
        features = torch.from_numpy(vals)
        neighb_inds = torch.from_numpy(neigh_inds)
        out = kpconv(q_pts, s_pts, neighb_inds, features)
        results[var] = out.detach().numpy()
    return results, grid


def demo():
    ds = WeatherDataset(sample_size=64, sample_jitter=32, length=3)
    batch = [ds[i] for i in range(3)]
    outputs, grid = run_batch(batch)
    for var, out in outputs.items():
        print(f"{var}: output shape {out.shape}")
    print("kernel grid points:", len(grid))


if __name__ == "__main__":
    demo()

import numpy as np
from torch.utils.data import Dataset

class WeatherDataset(Dataset):
    """Synthetic weather dataset producing random sensor points.

    Each sample returns a random subset of points for every variable. The number
    of returned points can vary within ``sample_jitter`` of ``sample_size``.
    """

    def __init__(self, variables=("u10", "v10", "t2", "ro"), points_per_var=1024,
                 sample_size=128, sample_jitter=0, length=1000):
        """Initialize random data for each variable.

        Parameters
        ----------
        variables: sequence of str
            Names of weather variables.
        points_per_var: int
            Number of points generated for each variable.
        sample_size: int
            Base number of points sampled for every item and variable.
        sample_jitter: int
            Random variation applied to ``sample_size`` for every variable.
        length: int
            Number of items in the dataset.
        """
        self.variables = list(variables)
        self.points_per_var = points_per_var
        self.sample_size = sample_size
        self.sample_jitter = sample_jitter
        self.length = length
        # Pre-generate random coordinates and values for every variable
        self._data = {}
        for var in self.variables:
            self._data[var] = {
                "points": np.random.rand(points_per_var, 3).astype(np.float32),
                "values": np.random.randn(points_per_var).astype(np.float32),
            }

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        sample = {}
        for var in self.variables:
            coords = self._data[var]["points"]
            vals = self._data[var]["values"]
            n = self.sample_size
            if self.sample_jitter > 0:
                jitter = np.random.randint(-self.sample_jitter, self.sample_jitter + 1)
                n = max(1, min(n + jitter, len(coords)))
            inds = np.random.choice(len(coords), size=n, replace=False)
            sample[var] = {
                "points": coords[inds],
                "values": vals[inds],
            }
        return sample


def collate_weather(batch):
    """Merge a list of dataset samples into a batch."""
    all_vars = batch[0].keys()
    collated = {}
    for var in all_vars:
        pts = [b[var]["points"] for b in batch]
        vals = [b[var]["values"] for b in batch]
        collated[var] = {
            "points": np.stack(pts, axis=0),
            "values": np.stack(vals, axis=0),
        }
    # Compute common convolution grid for all variables
    grid, step = compute_common_kernel_grid(collated)
    collated["kernel_grid"] = grid
    collated["kernel_step"] = step
    return collated


def compute_common_kernel_grid(batch):
    """Return a shared convolution grid for all variables.

    The grid step is estimated from the average nearest-neighbour distance
    over the union of all variable points.
    """

    all_pts = np.concatenate([v["points"].reshape(-1, 3) for v in batch.values()], axis=0)
    step = estimate_kernel_step(all_pts)
    min_b = all_pts.min(axis=0)
    max_b = all_pts.max(axis=0)
    xs = np.arange(min_b[0], max_b[0] + step, step)
    ys = np.arange(min_b[1], max_b[1] + step, step)
    zs = np.arange(min_b[2], max_b[2] + step, step)
    grid = np.stack(np.meshgrid(xs, ys, zs, indexing='ij'), axis=-1).reshape(-1, 3).astype(np.float32)
    return grid, float(step)


def estimate_kernel_step(points):
    if len(points) < 2:
        return 1.0
    diffs = points[:, None, :] - points[None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)
    dists[dists == 0] = np.inf
    min_d = dists.min(axis=1)
    return float(min_d.mean())

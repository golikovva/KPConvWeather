import numpy as np
from torch.utils.data import Dataset

class WeatherDataset(Dataset):
    """Synthetic weather dataset producing random sensor points."""

    def __init__(self, variables=("u10", "v10", "t2", "ro"), points_per_var=1024,
                 sample_size=128, length=1000):
        """Initialize random data for each variable.

        Parameters
        ----------
        variables: sequence of str
            Names of weather variables.
        points_per_var: int
            Number of points generated for each variable.
        sample_size: int
            Number of points sampled for every item and variable.
        length: int
            Number of items in the dataset.
        """
        self.variables = list(variables)
        self.points_per_var = points_per_var
        self.sample_size = sample_size
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
            inds = np.random.choice(len(coords), size=self.sample_size, replace=False)
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
    return collated

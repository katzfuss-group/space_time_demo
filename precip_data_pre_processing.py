# use this script to preprocess precipitation data
import pyreadr
import numpy as np
import torch
import math

def load_rdata(filepath):
    """Load RData file and return as a float32 NumPy array."""
    data = pyreadr.read_r(filepath)
    return data["all"].to_numpy().astype(np.float32)

def split_data(data, d=3):
    """Split the data into predictor (X) and response (Y) parts."""
    ns = data.shape[0] - d
    X = data[0:d, :]
    Y = data[d:d+ns, :]
    return X, Y

def lonlat_to_xyz(X):
    """Convert (lon, lat) to (x, y, z) coordinates on a sphere."""
    lon_2pi = X[0, :] / 360 * 2 * math.pi
    lat_2pi = X[1, :] / 360 * 2 * math.pi

    x = np.cos(lat_2pi) * np.cos(lon_2pi)
    y = np.cos(lat_2pi) * np.sin(lon_2pi)
    z = np.sin(lat_2pi)

    return torch.from_numpy(x), torch.from_numpy(y), torch.from_numpy(z)

def standardize_tensor(tensor):
    """Standardize a tensor to mean 0 and std 1."""
    return (tensor - torch.mean(tensor)) / torch.std(tensor)

def process_coordinates(X):
    """Convert and standardize spatial-temporal coordinates."""
    x_loc, y_loc, z_loc = lonlat_to_xyz(X)
    t_loc = torch.from_numpy(X[2, :].reshape(-1, 1))

    x_loc = standardize_tensor(x_loc)
    y_loc = standardize_tensor(y_loc)
    z_loc = standardize_tensor(z_loc)
    t_loc = standardize_tensor(t_loc)

    return x_loc, y_loc, z_loc, t_loc

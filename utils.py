import numpy as np
import torch
import random
import gpytorch
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
import cartopy.crs as ccrs
import math
from maxmin_exact import maxmin_exact

class MyKernel(gpytorch.kernels.Kernel):
    def __init__(self, nu: float = 2.5, **kwargs):
        super(MyKernel, self).__init__(**kwargs)
        self.register_parameter(name="raw_spacescale",
                                parameter=torch.nn.Parameter(torch.tensor(1.0)))
        self.register_parameter(name="raw_timescale",
                                parameter=torch.nn.Parameter(torch.tensor(1.0)))
        constraint = gpytorch.constraints.Positive()
        self.register_constraint("raw_spacescale", constraint)
        self.register_constraint("raw_timescale", constraint)
        self.base_kernel = gpytorch.kernels.MaternKernel(nu, ard_num_dims=1)
        self.base_kernel.lengthscale = 1.0
        for parm in self.base_kernel.parameters():
            parm.requires_grad = False

    @property
    def spacescale(self):
        return self.raw_spacescale_constraint.transform(self.raw_spacescale)

    @property
    def timescale(self):
        return self.raw_timescale_constraint.transform(self.raw_timescale)

    def forward(self, x1, x2, diag=False, **params):
        x1_scale = x1.detach().clone()
        x2_scale = x2.detach().clone()
        x1_scale[:, :3] = x1[:, :3] / self.spacescale
        x2_scale[:, :3] = x2[:, :3] / self.spacescale
        x1_scale[:, 3] = x1[:, 3] / self.timescale
        x2_scale[:, 3] = x2[:, 3] / self.timescale
        print('Calling from inside the forward function in mykernel..')

        return self.base_kernel.forward(x1_scale, x2_scale, diag, **params)


def sample_spatiotemporal_indices(X, Y, sample_size=5000, n_samp=5):
    """Randomly sample location indices and ensemble (year) indices."""
    loc_indices = np.random.randint(0, X.shape[1], size=sample_size)
    ensemble_indices = np.random.randint(0, Y.shape[0], size=n_samp)
    print(f"Ensemble used is: {ensemble_indices}")
    return loc_indices, ensemble_indices

def extract_sampled_locations(x_loc, y_loc, z_loc, t_loc, loc_indices):
    """Extract and reshape location values based on sampled indices."""
    x_picked = x_loc[loc_indices].reshape(-1, 1)
    y_picked = y_loc[loc_indices].reshape(-1, 1)
    z_picked = z_loc[loc_indices].reshape(-1, 1)
    t_picked = t_loc[loc_indices].reshape(-1, 1)

    # Combine spatial and temporal coordinates
    X_sampled = torch.tensor(np.concatenate((x_picked, y_picked, z_picked, t_picked), axis=1))
    return X_sampled

def extract_sampled_response(Y, ensemble_indices, loc_indices):
    """Extract response variable from the sampled ensemble and locations."""
    y_sampled = torch.tensor(Y[ensemble_indices][0, loc_indices])
    return y_sampled


# gp_training.py

def train_gp_model(kernel, X_sampled, y_sampled, sample_size, n_epoch=100, lr=1e-1):
    """Train GP model using Adam optimizer and return the kernel and NLL history."""
    print(f"lr is: {lr}")
    optimizer = torch.optim.Adam([{'params': kernel.parameters()}], lr=lr)

    for name, parm in kernel.named_parameters():
        print(f"Parameters in optimization are {name} with value {parm}")

    nll_ls = []

    for k in range(n_epoch):
        print(f'\nEpoch {k}')
        optimizer.zero_grad()

        cov_mat = kernel(X_sampled).evaluate() + torch.eye(sample_size) * 0.001
        cov_mat_inv = cov_mat.inverse()

        nll = (
            torch.logdet(cov_mat_inv) / 2 +
            torch.matmul(cov_mat_inv, y_sampled).dot(y_sampled) / 2
        )

        nll_ls.append(nll.item())
        nll.backward()
        optimizer.step()

        for name, parm in kernel.named_parameters():
            print(f"{name} = {parm.item()}, grad = {parm.grad}")


    print(f"space scale and time sclaes are: {kernel.spacescale.item()} and {kernel.timescale.item()}")
    return kernel.spacescale.item(), kernel.timescale.item()







def sample_and_train_gp(X, Y, x_loc, y_loc, z_loc, t_loc, sample_size=5000, n_samp=5, kernel_scale=1.5, n_epoch=100, lr=1e-1):
    # Sample indices
    loc_indices, ensemble_indices = sample_spatiotemporal_indices(X, Y, sample_size=sample_size, n_samp=n_samp)
    
    # Get sampled locations and responses
    X_sampled = extract_sampled_locations(x_loc, y_loc, z_loc, t_loc, loc_indices)
    y_sampled = extract_sampled_response(Y, ensemble_indices, loc_indices)

    # Initialize kernel
    kernel = MyKernel(kernel_scale)
    
    # Update sample size
    sample_size = X_sampled.shape[0]

    # Train the GP model
    space_scale_item, time_scale_item = train_gp_model(kernel, X_sampled, y_sampled, sample_size, n_epoch=n_epoch, lr=lr)

    return space_scale_item, time_scale_item


def avg_sample_and_train_gp(X, Y, x_loc, y_loc, z_loc, t_loc, sample_size=5000, n_samp=5, kernel_scale=1.5, n_epoch=100, lr=1e-1, num_samples=5):
    space_scales = []
    time_scales = []

    for _ in range(num_samples):
        space_scale_item, time_scale_item = sample_and_train_gp(X, Y, x_loc, y_loc, z_loc, t_loc, n_epoch=n_epoch, lr=lr, sample_size=sample_size, n_samp=n_samp, kernel_scale=kernel_scale)
        space_scales.append(space_scale_item)
        time_scales.append(time_scale_item)

    # Compute averages
    avg_space_scale = torch.mean(torch.tensor(space_scales))
    avg_time_scale = torch.mean(torch.tensor(time_scales))
    return avg_space_scale, avg_time_scale


def scale_inputs(x_loc, y_loc, z_loc, t_loc, space_scale_item, time_scale_item):
    x_loc = x_loc / space_scale_item
    y_loc = y_loc / space_scale_item
    z_loc = z_loc / space_scale_item
    t_loc = t_loc /time_scale_item
    return x_loc, y_loc, z_loc, t_loc

def scal_mean_sd(precip_train_n,precip_test):
    ## scale y train
    train_mean = precip_train_n.mean(axis=0)
    train_std = precip_train_n.std(axis=0)
    precip_train_n = (precip_train_n - train_mean) / train_std

    # scale y_test
    precip_test = (precip_test - train_mean) / train_std
    return precip_train_n,precip_test


def rev_ord(ord):
    rev_ord = np.zeros(ord.shape)
    rev_ord[ord] = np.arange(ord.shape[0])
    return rev_ord

def plot_seq_heatmap1(
        sample_reshape, FIGPATH="./", fig_name="test", suptitle="", nlat=190, nlon=288,
        numrow=1, numcol=5, vmin=-4.5, vmax=4.5, str_ints=None, row_labels=None, col_labels=None,
        show=False, extent=None, show_border=False, min_lat=-89.06, max_lat=89.06, min_lon=0, max_lon=360):
    """

    Generate and save a heatmap-like sequence of plots using given data with geographical context. This function
    creates visualizations using matplotlib and cartopy to display geographical data, with options for customization
    such as subplot configurations, data scaling, labels, color scaling, geographical extent, and borders.

    Parameters:
        sample_reshape (ndarray): A reshaped array of data, where each column corresponds to a time snapshot or
            data state to be visualized on separate subplots.
        FIGPATH (str): The path where the generated plot image will be saved. Defaults to "./".
        fig_name (str): The filename for the saved plot. Defaults to "test".
        suptitle (str): The title for the entire figure. Defaults to an empty string.
        nlat (int): Number of latitude points in the reshaped data. Defaults to 190.
        nlon (int): Number of longitude points in the reshaped data. Defaults to 288.
        numrow (int): Number of rows of subplots in the figure. Defaults to 1.
        numcol (int): Number of columns of subplots in the figure. Defaults to 5.
        vmin (float): Minimum value for color scaling. Defaults to -4.5.
        vmax (float): Maximum value for color scaling. Defaults to 4.5.
        str_ints (list of str): List of strings to annotate each subplot with specific textual information (e.g., time steps).
        row_labels (list of str): Labels for subplot rows, displayed along the left side.
        col_labels (list of str): Labels for subplot columns, displayed along the top side.
        show (bool): Whether to display the plots immediately. If False, the plot is saved to a file. Defaults to False.
        extent (tuple of float): The geographical bounds of the plot in the form (min_lon, max_lon, min_lat, max_lat).
            If provided, this value adjusts plot dimensions to ensure proportionality with the geographical region.
        show_border (bool): Whether or not to display geographical borders and coastline features. Defaults to False.
        min_lat (float): Minimum latitude for the data grid. Defaults to -89.06.
        max_lat (float): Maximum latitude for the data grid. Defaults to 89.06.
        min_lon (float): Minimum longitude for the data grid. Defaults to 0.
        max_lon (float): Maximum longitude for the data grid. Defaults to 360.

    Returns:
        None

    Raises:
        None
    """
    # fig, ax = plt.subplots(numrow, numcol, figsize=(numcol * 2, numrow * 2), subplot_kw={'projection': ccrs.PlateCarree()})

    # Base dimensions for a single subplot
    base_width, base_height = 2, 2  # Adjust as needed for subplot size

    # Adjust dimensions if extent is provided to match the scale
    if extent:
        lon_range = abs(extent[1] - extent[0]) / 360  # Proportion of the 360-degree range
        lat_range = abs(extent[3] - extent[2]) / 180  # Proportion of the 180-degree range
    else:
        lon_range, lat_range = 1, 1  # Full globe

    # Calculate total figure size dynamically
    fig_width = numcol * base_width * lon_range
    fig_height = numrow * base_height * lat_range
    fig, ax = plt.subplots(numrow, numcol, figsize=(fig_width, fig_height), subplot_kw={'projection': ccrs.PlateCarree()})

    # Reduce all spacing between subplots
    plt.subplots_adjust(wspace=0, hspace=0)

    lats = np.linspace(min_lat, max_lat, nlat)  # this might need to be updated since poles are removed
    lons = np.linspace(min_lon, max_lon, nlon)
    lon_grid, lat_grid = np.meshgrid(lons, lats)

    images = []
    for d in range(numrow * numcol):
        precs_fx_d = sample_reshape[:, d]
        reshape =np.reshape(np.ravel(precs_fx_d, order='F'), (nlat, nlon))
        x, y = divmod(d, numcol)
        ax_current = ax[y] if numrow == 1 else ax[x, y]

        # Plot data
        if extent:
            ax_current.set_extent(extent, crs=ccrs.PlateCarree())
            images.append(
                ax_current.pcolormesh(lon_grid, lat_grid, reshape, transform=ccrs.PlateCarree(), cmap='Spectral_r', vmin=vmin, vmax=vmax))
        elif show_border:
            images.append(
                ax_current.pcolormesh(lon_grid, lat_grid, reshape, transform=ccrs.PlateCarree(), cmap='Spectral_r', vmin=vmin, vmax=vmax))
        else:
            images.append(ax_current.imshow(reshape, cmap='Spectral_r', vmin=vmin, vmax=vmax))

        # Add title and features
        if str_ints:
            ax_current.set_title(f"t = {str(str_ints[d])}")
        ax_current.set_xticks([]), ax_current.set_yticks([])
        if show_border or extent:
            ax_current.add_feature(cfeature.BORDERS, linestyle='-', edgecolor='black')
            ax_current.add_feature(cfeature.COASTLINE, edgecolor='black')

        # Add row and column labels
        if row_labels and y == 0:
            ax_current.text(-0.3, 0.5, row_labels[x], va='center', ha='right', transform=ax_current.transAxes,
                            fontsize=15)
        if col_labels and x == 0:
            ax_current.text(0.5, 1.2, col_labels[y], va='bottom', ha='center', transform=ax_current.transAxes,
                            fontsize=15)

    plt.subplots_adjust(bottom=0.15, top=0.85)
    fig.colorbar(images[0], ax=ax, orientation='horizontal', fraction=.05, pad=0.05)
    fig.suptitle(suptitle)
    fig.set_rasterized(True)

    if show:
        plt.show()
    else:
        plt.savefig(FIGPATH + fig_name, dpi=1200, bbox_inches='tight', pad_inches=0)
        plt.close()




def plot_seq_heatmap(sample_reshape, FIGPATH, fig_name, suptitle, global_min, global_max):
    """

        :param data: 2d torch tensor
        :param FIGPATH: string of file path
        :param fig_name: string of fig name
        :param suptitle: string of suptitle
        :param global_min: float, min for vmin in imshow
        :param global_max: float, max for vmax in imshow
        :return: figure in .png
        """

    numrow = 3
    numcol = 10
    fig, ax = plt.subplots(numrow, numcol)
    # vmin = torch.min(sample_reshape)
    # vmax = torch.max(sample_reshape)

    for d in range(30):
        precs_fx_d = sample_reshape[:, d]
        reshape = np.reshape(np.ravel(precs_fx_d, order='F'), (74, 37))
        x = math.floor(d / numcol)
        y = d % numcol
        im0 = ax[x, y].imshow(reshape, cmap='Spectral_r', vmin= global_min, vmax= global_max)
        # ax[x, y].imshow(reshape, cmap='Spectral_r', vmin=global_min, vmax=global_max)
        ax[x, y].set_xticks([])
        ax[x, y].set_yticks([])

    plt.colorbar(im0, ax=ax.ravel().tolist(), shrink = 0.3, anchor = (1.0,1.0))

    fig.suptitle(suptitle)
    plt.savefig(FIGPATH + fig_name, dpi=600)
    plt.close()


def time_ordering(locs,train_time=30,sead = 123, n_locs=2738):
    locs_spatial = locs[0:n_locs,:]
    np.random.seed(sead)
    spatial_ord = maxmin_exact(locs_spatial)
    ord_combine = spatial_ord
    for i in range(train_time):
        ord_combine = np.concatenate((ord_combine, spatial_ord + i * len(spatial_ord)), axis=None)

    ord_combine = ord_combine[len(spatial_ord):]
    return ord_combine
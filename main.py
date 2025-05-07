# -------------------------------------------------------------
# import required packages
# -------------------------------------------------------------
import numpy as np
import time
import torch
from sklearn.model_selection import train_test_split
import precip_data_pre_processing as prep
import utils
from maxmin_exact import maxmin_exact
from NN_L2 import NN_L2
from  batram.legmods import compute_scale
import batram.legmods as bat


# -------------------------------------------------------------
# read data and data processing
# -------------------------------------------------------------


data = prep.load_rdata("data/prec_days_101_81k_10_20.RData")
locs, precs = prep.split_data(data, d=3)
precs = np.log(precs + 1e-10) # log transform
x_loc, y_loc, z_loc, t_loc = prep.process_coordinates(locs)
Ns, Nt = 2738, 30 # number of spatial locations in each time frame and number of time frame
nlat, nlon = 74, 37
min_lat, max_lat, min_lon, max_lon = -29.68, 39.11, 250, 295
# -------------------------------------------------------------
# ranges pretrain
# -------------------------------------------------------------

# # train GP on samples
# tic = time.perf_counter()
# space_scale_item, time_scale_item = utils.avg_sample_and_train_gp(locs, precs, x_loc, y_loc, z_loc, t_loc,n_epoch=100,n_samp=5)
# # scale inputs
# x_loc, y_loc, z_loc, t_loc = utils.scale_inputs(x_loc, y_loc, z_loc, t_loc, space_scale_item, time_scale_item)
# print(f"range estimation used {time.perf_counter() - tic:0.4f} seconds")
# print(f"space scale is: {space_scale_item}, time scale is: {time_scale_item}")
x_loc, y_loc, z_loc, t_loc = utils.scale_inputs(x_loc, y_loc, z_loc, t_loc, 0.039, 0.044)

# -------------------------------------------------------------
# ordering and neighbor selection with maximin ordering
# -------------------------------------------------------------
locs = np.transpose(locs)
# odr = maxmin_exact(locs) # maximin ordering
odr = utils.time_ordering(locs) # time ordering

locs = locs[odr, :]
precs = precs[:, odr]
m = 30
NN = NN_L2(locs, m)
NN = torch.from_numpy(NN)[:, 1:]  # ignore the location itself
scal = compute_scale(locs, NN)

# -------------------------------------------------------------
# model training
# -------------------------------------------------------------
## split training and testing data
# Split data into training and testing sets
precs_train, precs_test = train_test_split(precs, test_size=0.8, random_state=42)
print(f"Number of training samples: {precs_train.shape[0]}")
precs_train, precs_test = utils.scal_mean_sd(precs_train, precs_test)
precs_train = torch.from_numpy(precs_train)
train_data = bat.Data.new(locs, precs_train.float(), NN)

thetaInit = torch.tensor([2.24, 3.01, -0.96, 1.09, 0.47, -2.21 ])  # use this pretrained theta init to check log_prob


tm = bat.SimpleTM(train_data, thetaInit, False, smooth=1.5, nugMult=4.0)
maxIter, lr, batch_size = 30, 1e-3, 1000
tic = time.perf_counter()
res = tm.fit(maxIter, init_lr=lr, batch_size=batch_size)
print(f"fit_map used {time.perf_counter() - tic:0.4f} seconds")


# -------------------------------------------------------------
# draw samples
# -------------------------------------------------------------
tic = time.perf_counter()
# new_sample = tm.cond_sample() # whole sample
print(f"draw sample used {time.perf_counter() - tic:0.4f} seconds")

# conditional sampling
con_t = 10 # conditioned on the first ten time frames
partial_field = precs_test[0, :(int(Ns) * con_t)]
partial_field = torch.from_numpy(partial_field).float().squeeze()
new_sample = tm.cond_sample(xFix=partial_field) # conditional sampling for time ordering

# print estimated thetas
print(f"estimated thetas are: {res.parameters.get('theta.theta')}")

# -------------------------------------------------------------
# plot drawn samples
# -------------------------------------------------------------
rev_ord = utils.rev_ord(odr)
new_sample = new_sample[:, rev_ord]

utils.plot_seq_heatmap(
    torch.t(torch.reshape(new_sample, (Nt, Ns))), FIGPATH="./", fig_name="sample", suptitle="", nlat=nlat, nlon=nlon,
    numrow=3, numcol=10, vmin=-4.5, vmax=4.5, str_ints=None,
    row_labels=None, col_labels=None,
    show=True, extent=(min_lon, max_lon, min_lat, max_lat), show_border=True, min_lat=min_lat, max_lat=max_lat,
    min_lon=min_lon,
    max_lon=max_lon)

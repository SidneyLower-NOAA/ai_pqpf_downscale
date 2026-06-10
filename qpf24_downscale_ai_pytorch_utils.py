import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
import grib2io
from scipy import signal, ndimage
import os
from contextlib import contextmanager, redirect_stdout

### ------------------------- ###
###  Process Data for Input
### -------------------------- ###

@contextmanager
def suppress_stdout():
    # Open devnull to send output into the void
    with open(os.devnull, 'w') as devnull:
        old_stdout_fd = os.dup(1)
        try:
            # Redirect Python's stdout
            with redirect_stdout(devnull):
                # Redirect stdout to devnull
                os.dup2(devnull.fileno(), 1)
                yield
        finally:
            # Restore the original file descriptor
            os.dup2(old_stdout_fd, 1)
            os.close(old_stdout_fd)

def load_constants(FIXblend: str):

    FIXai = FIXblend+"/AI/precip/"
    
    # stats for normalizing wrt training data                                                                                                       
    qpe_stats = xr.open_dataset(
           FIXai+"downscaling_training_stats_for_normalization.nc"
        ).load()
    precip_mean = qpe_stats.mean_log_precip.values
    precip_std = qpe_stats.std_log_precip.values

    # Process LOW RES terrain features                                                                                                              
    terrain_file = (
        FIXai+"terrain_20km_nml_to.nc"
    )
    ds_topo_data = xr.open_dataset(terrain_file).load()
    # regrid onto 2.5km grid                                                                                                                        
    with suppress_stdout():
        grid2p5=grib2io.open(f'{FIXai}/hiresw.t00z.arw_2p5km_one_message.grib2')
        grid20=grib2io.open(f'{FIXai}/hiresw.t00z.fv3_20km_one_message.grib2')
        grid_out2p5 = grib2io.Grib2GridDef.from_section3(grid2p5[0].section3)
        ds_topo_data.nml_terrain_20km.attrs["GRIB2IO_section3"] = grid20[0].section3
        interp_terrain = ds_topo_data.nml_terrain_20km.grib2io.interp("budget", grid_out2p5, num_threads=1)

    terrain_20km = np.nan_to_num(interp_terrain.values)

    # Process HIGH RES terrain features
    terrain_file = (
        FIXai+"terrain_2p5km_nml_to.nc"
    )
    ds_topo_data = xr.open_dataset(terrain_file).load()
    terrain_2p5km = np.nan_to_num(ds_topo_data.nml_terrain_2p5km.values)

    return precip_mean, precip_std, terrain_20km, terrain_2p5km



def load_qpf_data(data_path: str):

    ds = xr.open_dataset(data_path, decode_timedelta=True, engine='zarr')
    da = ds.pqpf24_percentile_prediction

    return da

def feather_cliff_edges(ai_data, threshold=0.254, decay_rate=0.02):
    """
    Finds the blunt cliff edges in the AI PQPF data smooths them out with an exponential 
    physical decay tail matching the MRMS edge topology.
    """
    rain_mask = ai_data > threshold
    distance_map = ndimage.distance_transform_edt(rain_mask)
    feather_multiplier = 1.0 - np.exp(-distance_map * decay_rate)
    feathered_data = ai_data * feather_multiplier
    feathered_data[~rain_mask] = 0.0
    
    return feathered_data

            
class xr_to_tensor(torch.utils.data.Dataset):
    def __init__(self, percentile_data: xr.DataArray,
                 qpe_stats_mean: np.array, qpe_stats_std: np.array,
                 terrain_20km: np.array, terrain_2p5km: np.array):

        
        self.da = percentile_data

        # the way this will run is each lead time has its own task
        # BUT we want to process all percentiles, so n_samples
        # will be the number of percentiles
        percentiles = percentile_data.percentiles.values
            
        self.n_samples = len(percentiles)
        self.percentiles = percentiles

        self.precip_mean = qpe_stats_mean
        self.precip_std = qpe_stats_std

        lowres_terrain = torch.from_numpy(np.nan_to_num(terrain_20km)).float()
        highres_terrain = torch.from_numpy(np.nan_to_num(terrain_2p5km)).float()

        # Compute elevation gradient, difference between 20km and 2.5km resolution
        self.elev_diff = (highres_terrain - lowres_terrain).unsqueeze(0)
        self.highres_features = highres_terrain.unsqueeze(0)
        
    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):


        percentile = self.percentiles[idx]

        interp20_to_2p5 = np.nan_to_num(self.da.isel(percentiles=idx).values, 0.0)
        valid_date = pd.to_datetime(self.da.validDate.values)

        logp1_feature = np.log1p(feather_cliff_edges(interp20_to_2p5))
        normalized_feature = (logp1_feature - self.precip_mean) / self.precip_std

        # add timing tensors
        day_of_year = valid_date.day_of_year
        ending_hour = int(valid_date.hour)
        days_in_year = 366.0 if valid_date.is_leap_year else 365.0
        sin_time = np.sin(2 * np.pi * day_of_year / days_in_year)
        cos_time = np.cos(2 * np.pi * day_of_year / days_in_year)
        ending_hour_sin = np.sin(2 * np.pi * (ending_hour / 24.0))
        ending_hour_cos = np.cos(2 * np.pi * (ending_hour / 24.0))

        # to be injected with the FiLM Layers [4, 1]
        time_vector = torch.tensor(
            [sin_time, cos_time, ending_hour_sin, ending_hour_cos], dtype=torch.float32
        )
        # shape [3, 2.5km H, 2.5km W]
        feature_tensor = torch.tensor(
            normalized_feature, dtype=torch.float32
        ).unsqueeze(0)
        combined_features = torch.cat(
            [feature_tensor, self.highres_features, self.elev_diff], dim=0
        )

        return combined_features, time_vector, interp20_to_2p5, percentile

# Write out to Zarr
def write_high_res_ds(
    hires_output, percentiles, latitude, longitude, ref_date, lead_time, output_file,
):

    sort_idx = percentiles.argsort()

    # deal with datetime stuff
    leadTime = pd.Timedelta(hours=lead_time)
    refDate = ref_date
    validDate = ref_date + leadTime

    # sort data arrays
    output_sorted = hires_output[sort_idx]
    percentiles_sorted = percentiles[sort_idx]

    output_smoothed = np.zeros_like(output_sorted)
    for grid in range(len(percentiles)):
        output_smoothed[grid] = sgolay2d(output_sorted[grid], 11, 3)
    
    # write out
    da = xr.DataArray(
                    data=output_smoothed,
                    dims=["percentiles", "y", "x"],
                    coords=dict(
                            refDate=refDate,
                            validDate=validDate,
                            leadTime=leadTime,
                            duration=pd.Timedelta(hours=24),
                            latitude=(["y", "x"],latitude),
                            longitude=(["y", "x"],longitude),
                            percentiles=(['percentiles'], percentiles_sorted)
                        )
                    )

    da.name = 'pqpf24_percentile_prediction'
    da.to_zarr(output_file, mode="w")
    return


# Smoothing (stolen from Eric) :)
def sgolay2d (z, window_size, order):
    """
    Apply a Savitsky-Golay filter to a 2D array.
    """
    # number of terms in the polynomial expression
    n_terms = ( order + 1 ) * ( order + 2)  / 2.0

    if  window_size % 2 == 0:
        raise ValueError('window_size must be odd')

    if window_size**2 < n_terms:
        raise ValueError('order is too high for the window size')

    half_size = window_size // 2

    # exponents of the polynomial.
    # p(x,y) = a0 + a1*x + a2*y + a3*x^2 + a4*y^2 + a5*x*y + ...
    # this line gives a list of two item tuple. Each tuple contains
    # the exponents of the k-th term. First element of tuple is for x
    # second element for y.
    # Ex. exps = [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), ...]
    exps = [ (k-n, n) for k in range(order+1) for n in range(k+1) ]

    # coordinates of points
    ind = np.arange(-half_size, half_size+1, dtype=np.float32)
    dx = np.repeat( ind, window_size )
    dy = np.tile( ind, [window_size, 1]).reshape(window_size**2, )

    # build matrix of system of equation
    A = np.empty( (window_size**2, len(exps)) )
    for i, exp in enumerate( exps ):
        A[:,i] = (dx**exp[0]) * (dy**exp[1])

    # pad input array with appropriate values at the four borders
    new_shape = z.shape[0] + 2*half_size, z.shape[1] + 2*half_size
    Z = np.zeros( (new_shape) )
    # top band
    band = z[0, :]
    Z[:half_size, half_size:-half_size] =  band -  np.abs( np.flipud( z[1:half_size+1, :] ) - band )
    # bottom band
    band = z[-1, :]
    Z[-half_size:, half_size:-half_size] = band  + np.abs( np.flipud( z[-half_size-1:-1, :] )  -band )
    # left band
    band = np.tile( z[:,0].reshape(-1,1), [1,half_size])
    Z[half_size:-half_size, :half_size] = band - np.abs( np.fliplr( z[:, 1:half_size+1] ) - band )
    # right band
    band = np.tile( z[:,-1].reshape(-1,1), [1,half_size] )
    Z[half_size:-half_size, -half_size:] =  band + np.abs( np.fliplr( z[:, -half_size-1:-1] ) - band )
    # central band
    Z[half_size:-half_size, half_size:-half_size] = z

    # top left corner
    band = z[0,0]
    Z[:half_size,:half_size] = band - np.abs( np.flipud(np.fliplr(z[1:half_size+1,1:half_size+1]) ) - band )
    # bottom right corner
    band = z[-1,-1]
    Z[-half_size:,-half_size:] = band + np.abs( np.flipud(np.fliplr(z[-half_size-1:-1,-half_size-1:-1]) ) - band )

    # top right corner
    band = Z[half_size,-half_size:]
    Z[:half_size,-half_size:] = band - np.abs( np.flipud(Z[half_size+1:2*half_size+1,-half_size:]) - band )
    # bottom left corner
    band = Z[-half_size:,half_size].reshape(-1,1)
    Z[-half_size:,:half_size] = band - np.abs( np.fliplr(Z[-half_size:, half_size+1:2*half_size+1]) - band )

    # solve system and convolve
    m = np.linalg.pinv(A)[0].reshape((window_size, -1))
    return signal.fftconvolve(Z, m, mode='valid')


"""
******* WARNING ********

DO NOT TOUCH ANYTHING BELOW THIS 
UNLESS YOU KNOW _EXACTLY_ WHAT YOU'RE DOING

:)
"""
### ------------------------- ###
###    Downscaling Model
### -------------------------- ###

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Time_Embedding(nn.Module):
    # transform time vector [day sin, day cos, hour sin, hour cos]
    # into dense embedding, imbuing non linearity

    def __init__(self, input_time_dim, time_embedding_dim):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(input_time_dim, time_embedding_dim // 2),
            nn.ReLU(),
            nn.Linear(time_embedding_dim // 2, time_embedding_dim),
            nn.ReLU(),
        )

    def forward(self, time_vector):
        return self.time_mlp(time_vector)


class FiLM_Layer(nn.Module):
    # https://arxiv.org/pdf/1709.07871
    # take as input a dense embedding representing our time vector (day/hour harmonics)
    # pass this through a linear layer to generate 2 vectors that describe our affine transformation
    # of the time info onto the feature map

    def __init__(self, num_features_in_layer, time_embedding_dim):
        super().__init__()
        # project time embedding into scale/shift params for this layer's channels
        self.projection = nn.Linear(time_embedding_dim, num_features_in_layer * 2)

    def forward(self, x, time_emb):
        # x: input tensor [Batch, Channels, Height, Width]
        # time_emb: dense representation of time vector [Batch, time_embedding_dim]

        params = self.projection(time_emb)

        # separate this tensor into 2 parameters: additive (shift) and multiplicative (scale)
        shift, scale = params.chunk(2, dim=1)

        # reshape for broadcasting over H and W
        shift = shift.unsqueeze(2).unsqueeze(3)
        scale = scale.unsqueeze(2).unsqueeze(3)

        # apply transformation to feature map: out = (1 + scale) * x + shift
        return (1 + scale) * x + shift


class FiLM_DoubleConv(nn.Module):

    # set up Conv sequence with FiLM layers
    # FiLM layer output channels == input channels in that layer

    def __init__(self, in_channels, out_channels, time_embedding_dim, kernel_size=3, dropout_factor=0.0):
        super().__init__()

        self.conv_batch_1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.film1 = FiLM_Layer(out_channels, time_embedding_dim) # self.film = same # of channels as out_channels
        self.film2 = FiLM_Layer(out_channels, time_embedding_dim)
        self.neuron_activation = nn.ReLU(inplace=True)
        self.conv_batch_2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        self.dropout = nn.Dropout2d(p=dropout_factor)

    def forward(self, x, time_emb):
        # first conv
        x = self.conv_batch_1(x)
        x = self.film1(x, time_emb)
        x = self.neuron_activation(x)

        #second
        x = self.conv_batch_2(x)
        x = self.film2(x, time_emb)
        x = self.neuron_activation(x)

        x = self.dropout(x)

        return x


class Downscale_Model(nn.Module):
    def __init__(
        self,
        grid_dims=(1597, 2345),
        in_channels=3,
        out_channels=1,
        features=64,
        n_conv_layers=3,
        kernel_size=3,
        input_time_dim=4,
        time_embedding_dim=128,
        pos_emb_dim=16,
        dropout_factor=0.0,
    ):
        super(Downscale_Model, self).__init__()

        self.ny, self.nx = grid_dims
        self.pos_emb = nn.Parameter(torch.randn(1, pos_emb_dim, self.ny, self.nx))
        self.time_emb = Time_Embedding(input_time_dim, time_embedding_dim)

        total_in_channels = in_channels + pos_emb_dim

        # https://github.com/twtygqyy/pytorch-SRResNet/blob/master/srresnet.py
        # https://github.com/tensorlayer/SRGAN/blob/master/srgan.py

        self.conv_input = DoubleConv(total_in_channels, features, kernel_size)

        self.conv_hidden = nn.ModuleList()
        layer = 0
        while layer <= n_conv_layers:
            self.conv_hidden.append(DoubleConv(features, features, kernel_size))
            #self.conv_hidden.append(FiLM_DoubleConv(features, features, time_embedding_dim, kernel_size, dropout_factor))
            layer += 1

        self.film_conv = FiLM_DoubleConv(features, features, time_embedding_dim, kernel_size, dropout_factor)

        self.final = nn.Sequential(
            nn.Conv2d(features, out_channels, kernel_size=1), nn.Softplus())

    def forward(self, input_data, time_vector):

        batch_size = input_data.shape[0]
        pos_emb = self.pos_emb.expand(batch_size, -1, -1, -1)
        time_emb = self.time_emb(time_vector)

        x = torch.cat([input_data, pos_emb], dim=1)


        # expand feature dimensions from 4 --> n features
        x = self.conv_input(x)

        # now do series of convolutions, keeping feature density the same
        for conv_block in self.conv_hidden:
            x = conv_block(x)

        # include valid date info
        x_refined = self.film_conv(x, time_emb)

        # return final activation map
        return self.final(x_refined)    



def init_model(**kwargs):
    return Downscale_Model(**kwargs)

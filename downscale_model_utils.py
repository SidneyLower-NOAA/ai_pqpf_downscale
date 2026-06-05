import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
import os
from scipy import signal

### ------------------------- ###
###  Process Data for Input
### -------------------------- ###

class load_qpf_data(torch.utils.data.Dataset):
    def __init__(self, data_path: str, percentiles: np.array, data_format: str):

        self.data_file = data_path
        self.data_format = data_format
        self.ds = None

        # the way this will run is each lead time has its own task
        # BUT we want to process all percentiles, so n_samples
        # will be the number of percentiles
        self.n_samples = len(percentiles)
        self.percentiles = percentiles

        utils_path=os.environ["USHblend"]+"/downscale_model/"
        # stats for normalizing wrt training data
        qpe_stats = xr.open_dataset(
            utils_path+"downscaling_training_stats.nc"
        ).load()
        self.precip_mean = qpe_stats.mean_log_precip.values
        self.precip_std = qpe_stats.std_log_precip.values

        # Process LOW RES terrain features
        terrain_file = (
            utils_path+"upsampled_terrain20km.nc"
        )
        ds_topo_data = xr.open_dataset(terrain_file).load()
        terrain_tensor = (
            torch.from_numpy(np.nan_to_num(ds_topo_data.terrain20km.values, 0.0))
            .float()
            .unsqueeze(0)
        )
        lowres_features = terrain_tensor  # torch.cat([terrain_tensor], dim=0)

        # Process HIGH RES terrain features
        terrain_file = utils_path+"terrain_2p5km_nml_to.nc"
        ds_topo_data = xr.open_dataset(terrain_file).load()
        terrain_tensor = (
            torch.from_numpy(np.nan_to_num(ds_topo_data.nml_terrain_2p5km.values, 0.0))
            .float()
            .unsqueeze(0)
        )
        self.highres_terrain = terrain_tensor

        # Compute elevation gradient, difference between 20km and 2.5km resolution
        self.elev_diff = self.highres_terrain - lowres_features

    def __len__(self):
        return self.n_samples

    def _get_ds(self):
        # since we're only opening 1 file but iterating over percentiles
        # we can save time by opening it and loading it just once
        if self.ds is None:
            if self.data_format == 'grib2':
                ds = xr.open_dataset(self.data_file, engine='grib2io')
            elif self.data_format == 'zarr':
                ds = xr.open_dataset(self.data_file, decode_timedelta=True, engine='zarr')
            # on first time opening, load valid dates for quick look up later
            ds.validDate.load()
            self.ds = ds
        return self.ds


    def __getitem__(self, idx):

        self.ds = self._get_ds()
        percentile = self.percentiles[idx]

        # this is annoying
        if self.data_format == 'grib2':
            interp20_to_2p5 = np.nan_to_num(self.ds.APCP.isel(percentileValue=idx).values, 0.0)
        elif self.data_format == 'zarr':
            interp20_to_2p5 = np.nan_to_num(self.ds.pqpf24_percentile_prediction.isel(percentiles=idx).values, 0.0)

        valid_date = pd.to_datetime(self.ds.validDate.values)

        logp1_feature = np.log1p(interp20_to_2p5)
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
            [feature_tensor, self.highres_terrain, self.elev_diff], dim=0
        )

        return combined_features, time_vector, interp20_to_2p5, percentile

### ------------------------- ###
###    Smoothing (stolen from Eric) :)
### -------------------------- ###

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

### ------------------------- ###
###  Write out to Zarr
### -------------------------- ###

def write_high_res_ds(
    hires_output, percentiles, latitude, longitude, ref_date, lead_time, output_file, SMOOTHING,
):

    sort_idx = percentiles.argsort()

    # deal with datetime stuff
    leadTime = pd.Timedelta(hours=lead_time)
    refDate = ref_date
    validDate = ref_date + leadTime

    # sort data arrays
    output_sorted = hires_output[sort_idx]
    percentiles_sorted = percentiles[sort_idx]

    # OPTIONAL SMOOTHING
    if SMOOTHING:
        print(" ***** APPLYING SG SMOOTHING TO OUTPUT ***** ")
        output_smoothed = np.zeros_like(output_sorted)
        for grid in range(len(percentiles)):
            output_smoothed[grid] = sgolay2d(output_sorted[grid], 25, 3)
    else:
        output_smoothed = output_sorted

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
    print(f"... Finished writing Zarr: {output_file}")
    return



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


class FiLM_Refine(nn.Module):
    # set up Conv sequence with FiLM layers
    # FiLM layer output channels == input channels in that layer

    def __init__(
        self,
        in_channels,
        out_channels,
        time_embedding_dim,
        kernel_size=3,
        dropout_factor=0.0,
    ):
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False
        )

        self.film = FiLM_Layer(
            out_channels, time_embedding_dim
        )  # self.film = same # of channels as out_channels

        self.neuron_activation = nn.ReLU(inplace=True)

        self.conv_2 = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, padding=1, bias=False
        )

        self.dropout = nn.Dropout2d(p=dropout_factor)

        self.shuffle = nn.PixelShuffle(upscale_factor=2)

    def forward(self, x, time_emb):
        # first conv
        x = self.conv_1(x)
        x = self.film(x, time_emb)
        x = self.shuffle(x)
        x = self.neuron_activation(x)

        # second
        x = self.conv_2(x)
        x = self.film(x, time_emb)
        x = self.shuffle(x)
        x = self.neuron_activation(x)

        x = self.dropout(x)

        return x


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
            layer += 1

        # now pixel shuffle to as close as we can get to CONUS grid size == 3 upscalings
        # pixel shuffle works by decreasing channels while increasing image size
        #
        # C_out = C_in / upscale_factor^2
        # H_out = H_in * upscale_factor
        # W_out = W_in * upscale_factor

        # refine block is 2x conv --> insert time embeddings --> pixel shuffle --> activate

        self.refine = FiLM_Refine(
            features,
            256,
            time_embedding_dim,
            kernel_size=kernel_size,
            dropout_factor=dropout_factor,
        )

        # finally, to get to NBM CONUS shape, need to use Upsample, which interpolates to arbitrary shape
        self.full_size = nn.Upsample(
            size=(self.ny, self.nx), mode="bicubic", align_corners=False
        )

        # Transform from feature map space back to QPF
        # Use softplus activation function to ensure prediction is always >= 0
        # since target = log1p(ratio)
        self.final = nn.Sequential(
            nn.Conv2d(features, out_channels, kernel_size=1), nn.Softplus()
        )

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

        # perform pixel shuffles
        x_refined = self.refine(x, time_emb)

        # upsample to final grid size
        x_full_size = self.full_size(x_refined)

        return self.final(x_full_size)


def init_model(**kwargs):
    return Downscale_Model(**kwargs)

import pandas as pd
import numpy as np
import xarray as xr
from scipy import signal

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import sys
import os

from qpf24_downscale_ai_pytorch_utils import load_qpf_data, init_model





def worker_fn(rank: int, world_size: int, os_vars: list, collate_outputs: torch.tensor):

    FIXblend, DATA_IN, data_format, percentiles, (ny, nx), batch_size = os_vars
    
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    ### important! no multi-threading
    torch.set_num_threads(1)

    
    ### load trained model weights
    if rank == 1:
        print("... Loading downscale model and setting weights")
    model_name = 'PQPF_downscale_model_trained_state'
    saved_state = torch.load(f"{FIXblend}/AI/precip/{model_name}.pth", map_location='cpu')
    in_channels = saved_state['model_args']['in_channels']
    features = saved_state['model_args']['n_features_max']
    n_conv_layers = saved_state['model_args']['n_conv_layers']
    downscale_model = init_model(grid_dims=(ny, nx),in_channels=in_channels,features=features,n_conv_layers=n_conv_layers)
    downscale_model.load_state_dict(saved_state['model_state_dict'])

    ### process data
    if rank == 1:
        print("... Loading data")
    input_data_tensors = load_qpf_data(DATA_IN, percentiles=percentiles, data_format=data_format)
    sampler = DistributedSampler(input_data_tensors, num_replicas=world_size, rank=rank, shuffle=False)
    input_data_loader = torch.utils.data.DataLoader(input_data_tensors, batch_size=batch_size, sampler=sampler, num_workers=0)

    print(f"     RANK {rank} handling {len(input_data_loader)} percentiles")

    ### run inference
    downscale_model.eval() 
    with torch.no_grad(): 
        ncount = 0
        for low_res_input, time_vector, grid20, percentile in input_data_loader:

            output_batch = downscale_model(low_res_input, time_vector)

            per_list = percentile.tolist()
            for out in range(len(output_batch)):
                collate_outputs[per_list[out]] = torch.expm1(output_batch[out].squeeze(0).squeeze(0)) * grid20[out]
                ncount += 1
                print(f"     RANK {rank}: grid {ncount} / {len(input_data_loader)}")
            
    dist.destroy_process_group()

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

# Write out to Zarr
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
    return

if __name__ == "__main__":

    print("-" * 60)
    print(" BEGIN DOWNSCALE MODEL INFERENCE")
    print("-" * 60)

    # timeit
    s = pd.Timestamp.now()

    ### ------------------------- ###
    ###  Command Line Args: lead time, data files
    ### -------------------------- ###
    
    COMIN = os.environ.get("COMIN")
    DATA_OUT_DIR=os.getenv("COMOUT")
    FIXblend=os.getenv("FIXblend")
    FIXai = FIXblend+'/AI/precip/'
    PDY = os.environ.get("PDY")
    cyc = os.environ.get("cyc")
    refDate=pd.to_datetime(PDY+cyc, format='%Y%m%d%H')
    
    LEAD_TIME = int(sys.argv[1])
    
    # SG smoothing will be optional
    # User can ovveride
    if len(sys.argv) == 3:
        SMOOTHING = int(sys.argv[2])
        sm_v = 'YES'
    else:
        SMOOTHING = False
        sm_v = 'NO'

    # handle output formatting
    DATA_IN = f'{COMIN}/AI_percentile_predictions_pqpf24_{PDY}{cyc}_{LEAD_TIME}h_2layer_10cat_35epocs_early_stop_2p5km.zarr'
    filename = os.path.basename(DATA_IN).split('.')[0]
    data_format = os.path.basename(DATA_IN).split('.')[-1]
    output_file = DATA_OUT_DIR + f"{filename}_downscaled.{data_format}"
    # just in case
    if data_format not in ['grib2', 'zarr']:
        raise TypeError(f"Cannot process data of type {data_format}. Must be GRIB2 or Zarr")

    # set threads
    TOTAL_THREAD = int(os.environ["NTHREAD"]) - 2
    batch_size = 2
    model_threads = TOTAL_THREAD

    print(" SCRIPT ARGS: ")
    print(f"              LEAD TIME: {LEAD_TIME}")
    print(f"              REF DATE: {refDate}")
    print(f"              INPUT DATA: {DATA_IN}")
    print(f"              OUTPUT FILE: {output_file}")
    print(f"              SMOOTHING?: {sm_v}")
    print(f"              NCPUS FOR UNET: {model_threads}")
    
    # get 2.5km grid
    CONST_FILE=FIXblend+'/precip/blend.precip_const.co.2p5.nc'
    with xr.open_dataset(CONST_FILE) as temp:
        latitude = temp.latitude.values
        longitude = temp.longitude.values
    
    # runtime consts
    percentiles = np.array([5,10,20,25,30,40,50,60,70,75,80,90,95])
    ny, nx = np.shape(longitude) 
    os_vars = [FIXblend, DATA_IN, data_format, percentiles, (ny, nx), batch_size]

    # run model in parallel
    collate_outputs = torch.zeros((len(percentiles), 1, ny, nx))

    collate_outputs.share_memory_()

    print(" SENDING DATA TO MULTIPROCESSING WORKERS")
    mp.spawn(
        worker_fn,
        args=(model_threads,os_vars, collate_outputs),
        nprocs=model_threads,
        join=True
    )


    ###  Save to zarr
    print("... Saving data")
    write_high_res_ds(collate_outputs.squeeze(1).numpy(), percentiles, latitude, longitude, refDate, LEAD_TIME, output_file, SMOOTHING)
    print(f"... Finished writing Zarr: {output_file}")

    
    f = pd.Timestamp.now()
    delt = ((f - s).total_seconds()) / 60.
    print("-" * 60)
    print(f"   Total inference time: {delt:.2f} minutes")
    print("-" * 60)





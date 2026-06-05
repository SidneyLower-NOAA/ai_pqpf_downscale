import pandas as pd
import numpy as np
import xarray as xr
import torch
from scipy import signal
import sys
import os

from qpf24_downscale_ai_pytorch_utils import load_qpf_data, init_model

def run_downscale_model(input_tensors, initialized_downscale_model, batch_size=2, io_threads=4):

    collate_outputs = np.zeros((len(input_data_tensors), ny, nx)) # n percentiles, hires output, ny, nx
    collate_percentiles = []

    input_data_loader = torch.utils.data.DataLoader(
        input_data_tensors,
        batch_size=batch_size,
        shuffle=False,
        num_workers=io_threads
    )

    
    downscale_model.eval() 
    with torch.no_grad(): 
        ncount = 0
        for low_res_input, time_vector, grid20, percentile in input_data_loader:
            
            output_batch = initialized_downscale_model(low_res_input, time_vector)
    
            collate_percentiles.extend(percentile.tolist())
            for out in range(len(output_batch)):
                collate_outputs[ncount] = np.expm1(output_batch[out].squeeze(0).squeeze(0).numpy()) * grid20[out].numpy()
                ncount += 1
            batch_count = ncount // 2
            print(f"     ... batch {batch_count} / {len(input_data_loader)}")

    return collate_outputs, np.array(collate_percentiles)



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

    # timeit
    s = pd.Timestamp.now()

    ### ------------------------- ###
    ###  Command Line Args: lead time, data files
    ### -------------------------- ###
    
    COMIN = os.environ.get("COMIN")
    COMOUT=os.getenv("COMOUT")
    FIXblend=os.getenv("FIXblend")
    FIXai = FIXblend+'/AI/precip/'
    PDY = os.environ.get("PDY")
    cyc = os.environ.get("cyc")
    
    LEAD_TIME = int(sys.argv[1])
    
    DATA_IN = f'{COMIN}/AI_percentile_predictions_pqpf24_{PDY}{cyc}_{LEAD_TIME}h_2layer_10cat_35epocs_early_stop_2p5km.zarr'
    DATA_OUT_DIR = COMOUT
    
    # SG smoothing will be optional
    # User can ovveride
    if len(sys.argv) == 3:
        SMOOTHING = int(sys.argv[2])
        sm_v = 'YES'
    else:
        SMOOTHING = False
        sm_v = 'NO'
    
    # I'm going to assume the job card handles the directory creation, etc.
    filename = os.path.basename(DATA_IN).split('.')[0]
    data_format = os.path.basename(DATA_IN).split('.')[-1]
    output_file = DATA_OUT_DIR + f"{filename}_downscaled.{data_format}"
    # just in case
    if data_format not in ['grib2', 'zarr']:
        raise TypeError(f"Cannot process data of type {data_format}. Must be GRIB2 or Zarr")
    
    
    print("-" * 60)
    print(" BEGIN DOWNSCALE MODEL INFERENCE")
    print("-" * 60)
    print(" SCRIPT ARGS: ")
    print(f"              LEAD TIME: {LEAD_TIME}")
    print(f"              REF DATE: {pd.to_datetime(PDY, format='%Y%m%d%H')}")
    print(f"              INPUT DATA: {DATA_IN}")
    print(f"              OUTPUT FILE: {output_file}")
    print(f"              SMOOTHING?: {sm_v}")
    
    # fixed vars for now
    TOTAL_THREAD = int(os.environ["NTHREAD"]) - 2
    batch_size = 2
    io_threads = TOTAL_THREAD // 4
    model_threads = TOTAL_THREAD - io_threads
    
    torch.set_num_threads(model_threads)
    print(f"              THREADS: {torch.get_num_threads()}")
    
    ### ------------------------- ###
    ###  Load & Process Data
    ### -------------------------- ###
    
    # hate to hard code, but i just can't be bothered to do something more sophisticated
    percentiles = np.array([5,10,20,25,30,40,50,60,70,75,80,90,95])
    
    print("... Loading data")
    print(f"       Percentiles: ({len(percentiles)}): {percentiles}")
    input_data_tensors = load_qpf_data(DATA_IN, percentiles=percentiles, data_format=data_format)
    
    ### ------------------------- ###
    ###  Initialize Model
    ### -------------------------- ###
    
    # get 2.5km grid
    CONST_FILE=FIXblend+'/precip/blend.precip_const.co.2p5.nc'
    with xr.open_dataset(CONST_FILE) as temp:
        latitude = temp.latitude.values
        longitude = temp.longitude.values
    ny, nx = np.shape(longitude) 
    
    print("... Loading model")
    ### load trained model weights
    model_name = 'PQPF_downscale_model_trained_state'
    saved_state = torch.load(f"{FIXai}/{model_name}.pth", map_location=torch.device('cpu'))
    
    in_channels = saved_state['model_args']['in_channels']
    features = saved_state['model_args']['n_features_max']
    n_conv_layers = saved_state['model_args']['n_conv_layers']
    
    downscale_model = init_model(grid_dims=(ny, nx),in_channels=in_channels,features=features,n_conv_layers=n_conv_layers)
    downscale_model.load_state_dict(saved_state['model_state_dict'])
    
    ### ------------------------- ###
    ###  Run Model
    ### -------------------------- ###
    print("... Running model")
    downscaled_outputs, sorted_percentiles = run_downscale_model(input_data_tensors, downscale_model, batch_size, io_threads)
            
    ### ------------------------- ###
    ###  Save to zarr
    ### -------------------------- ###
    ref_date = pd.to_datetime(PDY, format='%Y%m%d%H')
    print("... Saving data")
    write_high_res_ds(downscaled_outputs, sorted_percentiles, latitude, longitude, ref_date, LEAD_TIME, output_file, SMOOTHING)
    print(f"... Finished writing Zarr: {output_file}")

    
    f = pd.Timestamp.now()
    delt = ((f - s).total_seconds()) / 60.
    print("-" * 60)
    print(f"   Total inference time: {delt:.2f} minutes")
    print("-" * 60)





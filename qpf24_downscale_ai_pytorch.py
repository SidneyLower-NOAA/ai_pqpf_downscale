import pandas as pd
import numpy as np
import xarray as xr

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import sys
import os

from qpf24_downscale_ai_pytorch_utils import load_constants, load_qpf_data, xr_to_tensor, init_model, write_high_res_ds

# Force ALL underlying libraries to use exactly 1 thread per process
# Pytorch will make its own processes
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"


def worker_fn(rank: int, world_size: int, os_vars: list, para_vars: list, collate_outputs: torch.tensor):


    FIXblend, (ny, nx), percentiles, batch_size = os_vars
    percentile_da, nml_qpf_mean, nml_qpf_std, terrain_20km, terrain_2p5km = para_vars

    
    if rank == 1:
        print("... Initializing workers")
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    ### important! no multi-threading
    torch.set_num_threads(1)

    
    ### load trained model weights
    if rank == 1:
        print("... Loading downscale model and setting weights")
    #model_name = 'PQPF_downscale_model_trained_state'
    model_name = 'PQPF_downscale_model_trained_state'
    saved_state = torch.load(f"{FIXblend}/AI/precip/{model_name}.pth", map_location='cpu')
    in_channels = saved_state['model_args']['in_channels']
    features = saved_state['model_args']['n_features_max']
    n_conv_layers = saved_state['model_args']['n_conv_layers']
    grid_padding = saved_state['model_args']['grid_padding']
    downscale_model = init_model(grid_dims=(ny, nx),in_channels=in_channels,features=features,n_conv_layers=n_conv_layers,grid_padding=grid_padding)
    downscale_model.load_state_dict(saved_state['model_state_dict'])

    ### process data
    if rank == 1:
        print("... Loading data into pytorch")
    input_data_tensors = xr_to_tensor(percentile_da, nml_qpf_mean, nml_qpf_std,terrain_20km, terrain_2p5km, grid_padding=grid_padding)
    sampler = DistributedSampler(input_data_tensors, num_replicas=world_size, rank=rank, shuffle=False)
    input_data_loader = torch.utils.data.DataLoader(input_data_tensors, batch_size=batch_size, sampler=sampler, num_workers=0)

    ### run inference
    percentiles = percentiles.tolist()
    downscale_model.eval() 
    if rank == 1:
        print("... Starting inference")
    with torch.no_grad(): 
        ncount = 0
        for low_res_input, time_vector, grid20, percentile in input_data_loader:
            print(f"     Process {rank} starting")
            output_batch = downscale_model(low_res_input, time_vector)
            per_list = percentile.tolist()
            for out in range(len(output_batch)):
                print(f"     ... {per_list[out]}th percentile downscaled.")
                this_per = percentiles.index(per_list[out])
                # get QPF by multiplying downscale tensor by 20km resolution QPF
                # also crop to CONUS size to get rid of padded pixels
                collate_outputs[this_per] = torch.expm1(output_batch[out].squeeze(0).squeeze(0)[grid_padding:ny+grid_padding, :nx]) * grid20[out]
                ncount += 1

    dist.destroy_process_group()

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
    

    # handle output formatting
    DATA_IN = f'{COMIN}/AI_percentile_predictions_pqpf24_{PDY}{cyc}_{LEAD_TIME}h_2layer_10cat_35epocs_early_stop_2p5km.zarr'
    filename = os.path.basename(DATA_IN).split('.')[0]
    data_format = os.path.basename(DATA_IN).split('.')[-1]
    output_file = DATA_OUT_DIR + f"/{filename}_downscaled.{data_format}"
    # just in case
    if data_format not in ['grib2', 'zarr']:
        raise TypeError(f"Cannot process data of type {data_format}. Must be GRIB2 or Zarr")

    batch_size = 1
    model_threads = 13

    print(" SCRIPT ARGS: ")
    print(f"              LEAD TIME: {LEAD_TIME}")
    print(f"              REF DATE: {refDate}")
    print(f"              INPUT DATA: {DATA_IN}")
    print(f"              OUTPUT FILE: {output_file}")
    print(f"              NCPUS FOR UNET: {model_threads}")
    
    # get 2.5km grid
    CONST_FILE=FIXblend+'/precip/blend.precip_const.co.2p5.nc'
    with xr.open_dataset(CONST_FILE) as temp:
        latitude = temp.latitude.values
        longitude = temp.longitude.values
    
    # load data (constants + percentiles)
    percentiles = np.array([5,10,20,25,30,40,50,60,70,75,80,90,95])
    ny, nx = np.shape(longitude) 

    print("... Grabbing constants and percentile data")
    percentile_da = load_qpf_data(DATA_IN)
    nml_qpf_mean, nml_qpf_std, terrain_20km, terrain_2p5km = load_constants(FIXblend)
    os_vars = [FIXblend, (ny, nx), percentiles, batch_size]
    para_vars = [percentile_da, nml_qpf_mean, nml_qpf_std, terrain_20km, terrain_2p5km]

    # output
    collate_outputs = torch.zeros((len(percentiles), 1, ny, nx))
    
    
    # run model in parallel
    collate_outputs.share_memory_()

    mp.spawn(
        worker_fn,
        args=(model_threads,os_vars, para_vars, collate_outputs),
        nprocs=model_threads,
        join=True
    )

    ###  Save to zarr
    print("... Saving data")
    write_high_res_ds(collate_outputs.squeeze(1).numpy(), percentiles, latitude, longitude, refDate, LEAD_TIME, output_file)
    print(f"     writing to Zarr: {output_file}")

    
    f = pd.Timestamp.now()
    delt = ((f - s).total_seconds()) / 60.
    print("-" * 60)
    print(f"   Total inference time: {delt:.2f} minutes")
    print("-" * 60)





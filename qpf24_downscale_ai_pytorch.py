import pandas as pd
import numpy as np
import xarray as xr

import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

import sys
import os

from qpf24_downscale_ai_pytorch_utils import load_consts, load_qpf_data, xr_to_tensor, init_model, write_high_res_ds



def worker_fn(rank: int, world_size: int, os_vars: list, para_vars: list, collate_outputs: torch.tensor):


    FIXblend, (ny, nx), batch_size, data_format = os_vars
    percentile_da, nml_qpf_mean, nml_qpf_std, terrain_20km, terrain_2p5km = para_vars
    
    
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
    input_data_tensors = xr_to_tensor(percentile_da, nml_qpf_mean, nml_qpf_std,terrain_20km, terrain_2p5km)
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
                collate_outputs[per_list.index(out)] = torch.expm1(output_batch[out].squeeze(0).squeeze(0)) * grid20[out]
                ncount += 1
                print(f"     RANK {rank}: grid {ncount} / {len(input_data_loader)}")
            
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
    #TOTAL_THREAD = int(os.environ["NTHREAD"]) - 2
    batch_size = 2
    model_threads = 16

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


    # load data (constants + percentiles)
    percentile_da = load_qpf_data(DATA_IN, data_format)
    nml_qpf_mean, nml_qpf_std, terrain_20km, terrain_2p5km = load_consts(FIXblend)
    os_vars = [FIXblend, (ny, nx), batch_size]
    para_vars = [percentile_da, nml_qpf_mean, nml_qpf_std, terrain_20km, terrain_2p5km]

    # output
    collate_outputs = torch.zeros((len(percentiles), 1, ny, nx))
    
    
    # run model in parallel
    collate_outputs.share_memory_()

    print(" SENDING DATA TO MULTIPROCESSING WORKERS")
    mp.spawn(
        worker_fn,
        args=(model_threads,os_vars, para_vars, collate_outputs),
        nprocs=model_threads,
        join=True
    )

    print(f"[DEBUG]: output size: {collate_outputs.size()}")

    ###  Save to zarr
    #print("... Saving data")
    #write_high_res_ds(collate_outputs.squeeze(1).numpy(), percentiles, latitude, longitude, refDate, LEAD_TIME, output_file, SMOOTHING)
    #print(f"... Finished writing Zarr: {output_file}")

    
    f = pd.Timestamp.now()
    delt = ((f - s).total_seconds()) / 60.
    print("-" * 60)
    print(f"   Total inference time: {delt:.2f} minutes")
    print("-" * 60)





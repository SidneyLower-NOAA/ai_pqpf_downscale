import pandas as pd
import numpy as np
import xarray as xr
import torch
import sys
import os

from downscale_model_utils import load_qpf_data, init_model, write_high_res_ds

# timeit
s = pd.Timestamp.now()

### ------------------------- ###
###  Command Line Args: lead time, data files
### -------------------------- ###

LEAD_TIME = int(sys.argv[1])
PDY = sys.argv[2]
DATA_IN = sys.argv[3]
DATA_OUT_DIR = sys.argv[4]


# SG smoothing will be optional
# User can ovveride
if len(sys.argv) == 6:
    SMOOTHING = int(sys.argv[5])
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
TOTAL_THREAD = int(os.environ["NTHREAD"])
batch_size = 2
num_workers = TOTAL_THREAD - 12

torch.set_num_threads(12)
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
CONST_FILE=os.environ["FIXblend"]+'/precip/blend.precip_const.co.2p5.nc'
with xr.open_dataset(CONST_FILE) as temp:
    latitude = temp.latitude.values
    longitude = temp.longitude.values
ny, nx = np.shape(longitude) 

print("... Loading model")
### load trained model weights
model_name = 'PQPF_upsample_MRMS_dropout_smoothedPSD'
saved_state = torch.load(os.environ["USHblend"]+'/downscale_model/'+model_name+".pth", map_location=torch.device('cpu'))

in_channels = saved_state['model_args']['in_channels']
features = saved_state['model_args']['n_features_max']
n_conv_layers = saved_state['model_args']['n_conv_layers']

downscale_model = init_model(grid_dims=(ny, nx),in_channels=in_channels,features=features,n_conv_layers=n_conv_layers)
downscale_model.load_state_dict(saved_state['model_state_dict'])

### ------------------------- ###
###  Run Model
### -------------------------- ###

collate_outputs = np.zeros((len(input_data_tensors), ny, nx)) # n percentiles, hires output, ny, nx
collate_percentiles = []

input_data_loader = torch.utils.data.DataLoader(
        input_data_tensors,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

print("... Running model")
downscale_model.eval() 
with torch.no_grad(): 
    ncount = 0
    for low_res_input, time_vector, grid20, percentile in input_data_loader:
        
        output_batch = downscale_model(low_res_input, time_vector)

        collate_percentiles.extend(percentile.tolist())
        for out in range(len(output_batch)):
            collate_outputs[ncount] = np.expm1(output_batch[out].squeeze(0).squeeze(0).numpy()) * grid20[out].numpy()
            ncount += 1
        batch_count = ncount // 2
        print(f"     ... batch {batch_count} / {len(input_data_loader)}")
        
### ------------------------- ###
###  Save to zarr
### -------------------------- ###
ref_date = pd.to_datetime(PDY, format='%Y%m%d%H')
print("... Saving data")
write_high_res_ds(collate_outputs, np.array(collate_percentiles), latitude, longitude, ref_date, LEAD_TIME, output_file, SMOOTHING)


f = pd.Timestamp.now()
delt = ((f - s).total_seconds()) / 60.
print("-" * 60)
print(f"   Total inference time: {delt:.2f} minutes")
print("-" * 60)





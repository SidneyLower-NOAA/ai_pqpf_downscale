# AI PQPF Downscaling Model

UNet-like model built on Pytorch for mapping 20km GEFS/IFS AI PQPF output to 2.5km CONUS grid. 

## Python scripts

- qpf24_downscale_ai_pytorch.py: Runtime script, takes lead time as an argument with optional smoothing (default is no smoothing). Expects environment variables COMIN, COMOUT, PDY, and cyc. Also expects to find model state file (PQPF_downscale_model_trained_state) in FIXblend/AI/precip/

- qpf24_downscale_ai_pytorch_utils.py: Utilities script with data loading and UNet modules.


## Constants / FIX files

- downscaling_training_stats_for_normalization.nc: Log-transformed mean and standard deviation of training dataset for normalization

- terrain_X_nml_to.nc: Terrain elevation at 2.5km and 20km resolutions, normalized


## Running model inference

First, place files in fix_ai in $FIXblend/AI/precip/
Copy model state file (PQPF_downscale_model_trained_state) into same directory

To run stand alone (requires 1 node, 13 CPUs)

        ./run_single_lead_time.sh $LEAD_TIME


To run as part of AI PQPF post processing
    
        1. Place JBLEND_AI_QPF24_DOWNSCALE_JOB in blend/jobs
        2. Place 'qsub' command into blend lsf file after prediction/post-processing routine

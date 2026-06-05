Downscaling Model

    1. Place "downscale_model" directory in blend/ush.
    
        Contents:
        
        - *.nc: terrain constant files
        - PQPF_upsample_MRMS_dropout_smoothedPSD.pth: Downscale model state. The runtime script loads the model with this file
        - downscale_model_utils.py: Data processing and downscale model modules. 
        - apply_downscale_model.py: Runtime script. Takes args (lead time, PDY, data_in, data_out_directory, and optionally smoothing)


    2. To run stand alone

        ./run_single_lead_time.sh $LEAD_TIME


    3. To run as part of AI PQPF post processing
    
        1. Place JBLEND_AI_QPF24_DOWNSCALE_JOB in blend/jobs
        2. Replace or just point to the lsf file versions here

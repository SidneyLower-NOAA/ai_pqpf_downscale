#!/bin/bash

set -x
module reset

##### MODULE SETUP STOLEN FROM DAVE
module use /apps/dev/lmodules/intel/19.1.3.304
module load ve/nbm/5.1
module use /lfs/h3/mdl/nbm/save/apps/modulefiles
module load blend-utils/1.0


date
export PS4=' $SECONDS + '

####################################
# Specify NET Name and RUN
#
# RUN - RUN grib field to repack
####################################
export NET=${NET:-blend}
export RUN=${RUN:-blend}
export pgmout="OUTPUT.$$"

########################################
# SET THE EXECUTION VARIABLES           
########################################
export PTMPROOT=${PTMPROOT:-/lfs/h3/mdl/ptmp/david.rudack}
export STMPROOT=${STMPROOT:-/lfs/h3/mdl/stmp/david.rudack}
export COMOUTblendbase=${COMOUTblendbase:-${PTMPROOT}/blend/v5.1}
export COMINblendbase=${COMINblendbase:-${PTMPROOT}/blend/v5.1}
##export HOMEblend=/lfs/h3/mdl/nbm/noscrub/david.rudack/blend
export HOMEblend=/lfs/h3/mdl/nbm/noscrub/sidney.lower/blend
export SCRIPTSblend=${SCRIPTSblend:-${HOMEblend}/scripts}
export FIXblend=${FIXblend:-${HOMEblend}/fix}
export PARMblend=${PARMblend:-${HOMEblend}/parm}
export EXECblend=${EXECblend:-${HOMEblend}/exec}
export UTILblend=${UTILblend:-${HOMEblend}/ush/util}
export USHblend=${USHblend:-${HOMEblend}/ush}
########################################
# SET THE DATE VARIABLES           
########################################
export cyc="00"
export cycle=${cycle:-t${cyc}z}
setpdy.sh 12 1
. ./PDY
echo $PDY

LEAD_TIME=$1

export COMOUTblendbase=${COMOUTblendbase:-$(compath.py -o ${NET}/${blend_ver})}
export COMOUT=${COMOUT:-${COMOUTblendbase:?}}/blend.$PDY/$cyc/modeldata
mkdir -m 755 -p $COMOUT

export PBS_ACCT="NBM-DEV"
export pid=$$
export LOGblend=${PTMPROOT}/dailylog/blend/log.${PDY}/${cyc}
export job=DOWNSCALE_${LEAD_TIME}h_${cyc}.${pid}
export logfile=$LOGblend/${job}.out
export PBS_OUTPUTFILE=$logfile
export DATA_IN=${COMOUT}/AI_percentile_predictions_pqpf24_${PDY}${cyc}_${LEAD_TIME}h_2layer_10cat_35epocs_early_stop_2p5km.zarr

export COMOUT_TEST=/lfs/h3/mdl/nbm/noscrub/sidney.lower/blend/ush/downscale_model/test_output
mkdir $COMOUT_TEST


export NTHREAD=$NCPUS
cd $USHblend/downscale_model
python apply_downscale_model.py $LEAD_TIME $PDY$cyc $DATA_IN $COMOUT_TEST



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
export PDY=$(date -d "yesterday" '+%Y%m%d')
echo $PDY

LEAD_TIME=$1

export COMIN=${COMIN:-${COMINblendbase:?}}/blend.${PDY}/$cyc/modeldata
export COMOUTblendbase=${COMOUTblendbase:-$(compath.py -o ${NET}/${blend_ver})}
export COMOUT=${COMOUT:-${COMOUTblendbase:?}}/blend.$PDY/$cyc/modeldata
mkdir -m 755 -p $COMOUT

# override comout for testing purposes (aka don't save to dave's area)
export COMOUT_TEST=/lfs/h3/mdl/nbm/noscrub/sidney.lower/downscale_model_dev/test_output/
mkdir $COMOUT_TEST
export COMOUT=$COMOUT_TEST

export OMP_NUM_THREADS=13
#export OMP_PROC_BIND=true
#export OMP_PLACES=cores

#mpiexec -n 1 --ppn 1 --cpu-bind verbose,depth --depth $NTHREAD python $USHblend/qpf24_downscale_ai_pytorch.py $LEAD_TIME 
python $USHblend/qpf24_downscale_ai_pytorch.py $LEAD_TIME


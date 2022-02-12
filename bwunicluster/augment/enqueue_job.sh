#!/bin/bash
#SBATCH --mail-type=ALL
#SBATCH --mail-user=privat@claudiuskienle.de
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --time=02:00:00
#SBATCH --mem=180000mb
#SBATCH --export=ALL,EXECUTABLE="python -u ../../src/data_processing/augment_by_mask.py ../../resources/images/calibrated_masked/cropped ../../resources/images/calibrated_masked_augmented/cropped --jobs=2"
#SBATCH --output="augment.out"
#SBATCH -J AugData
#SBATCH --dependency=aftercorr:20466975


#Usually you should set
export KMP_AFFINITY=compact,1,0
#export KMP_AFFINITY=verbose,compact,1,0 prints messages concerning the supported affinity
#KMP_AFFINITY Description: https://software.intel.com/en-us/node/524790#KMP_AFFINITY_ENVIRONMENT_VARIABLE

export OMP_NUM_THREADS=$((${SLURM_JOB_CPUS_PER_NODE}/2))
echo "Executable ${EXECUTABLE} running on ${SLURM_JOB_CPUS_PER_NODE} cores with ${OMP_NUM_THREADS} threads"
startexe=${EXECUTABLE}
echo $startexe
exec $startexe

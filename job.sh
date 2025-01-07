#!/bin/bash
#SBATCH --job-name dqml
#SBATCH -D ./
#SBATCH -o ./slurm_output/output.%A_%a.out
#SBATCH --partition All
#SBATCH --array=[1,2,3,4,5]
#SBATCH --mail-user K.Izadi@campus.lmu.de
#SBATCH --mail-type ALL
echo Running on node $SLURMD_NODENAME at `date`
echo Git commit: $(git rev-parse HEAD)
. ./env/bin/activate
cd two_feature_app
python main.py $SLURM_ARRAY_TASK_ID $SLURM_ARRAY_JOB_ID$SLURM_ARRAY_TASK_ID
echo Finished at `date`

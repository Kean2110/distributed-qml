#!/bin/bash
#SBATCH --job-name dqml
#SBATCH -D ./
#SBATCH -o ./two_feature_app/output/%A/output.%A.out
#SBATCH --partition All
#SBATCH --array=[1,2,3,4]
#SBATCH --mail-user K.Izadi@campus.lmu.de
#SBATCH --mail-type ALL
echo Running on node $SLURMD_NODENAME at `date`
. ./env/bin/activate
cd two_feature_app
python main.py $SLURM_ARRAY_TASK_ID $SLURM_JOB_ID
echo Finished at `date`

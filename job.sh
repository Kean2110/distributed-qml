#!/bin/bash
#SBATCH --job-name dqml
#SBATCH -D ./
#SBATCH -o ./slurm_output/output.%A.out
#SBATCH --partition All
#SBATCH --mail-user K.Izadi@campus.lmu.de
#SBATCH --mail-type ALL
echo Running on node $SLURMD_NODENAME at `date`
. ./env/bin/activate
cd two_feature_app
python main.py 
echo Finished at `date`

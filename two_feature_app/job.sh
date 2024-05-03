#!/bin/bash
#SBATCH --job-name=dqml
#SBATCH --partition=AMD
#SBATCH --ntasks-per-node=1
#SBATCH --get-user-env
#SBATCH --requeue
echo Running on node $SLURMD_NODENAME at `date`
. ~/env/bin/activate
cd two_feature_app
python main.py 
echo Finished at `date`

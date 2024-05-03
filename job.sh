#!/bin/bash
#SBATCH --job-name=dqml
#SBATCH --mail-user=K.Izadi@campus.lmu.de
#SBATCH --mail-type=FAIL
#SBATCH --requeue
echo Running on node $SLURMD_NODENAME at `date`
. ~/env/bin/activate
cd two_feature_app
python main.py 
echo Finished at `date`

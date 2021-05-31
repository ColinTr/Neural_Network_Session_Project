#!/bin/bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=22000M
#SBATCH --gres=gpu:1
#SBATCH --time=00-12:00:00  # DD-HH:MM:SS

# Environnement python
module load python/3.7
cd $SLURM_TMPDIR
virtualenv --no-download env
source env/bin/activate

python --version

pip3 install --no-index -r ~/projects/def-sponsor00/$USER/projet_session_rn/requirements.txt

cd ~/projects/def-sponsor00/$USER/projet_session_rn/src/

python scripts/runner.py

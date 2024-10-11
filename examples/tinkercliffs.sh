#!/bin/bash

#SBATCH --job-name=generate-ceo-data
#SBATCH --account=qc_group
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=jfurches@vt.edu

#SBATCH --partition=normal_q
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=6-00:00:00
#SBATCH --export=ALL


module load site/tinkercliffs/easybuild/setup
module load Anaconda3/2020.11

REPO_DIR=$(git rev-parse --show-toplevel)
ENV=~/env/ceo-adapt-vqe

source activate $ENV

# By running source activate, if the enviroment is not found,
# it returns a non-zero exit code which is stored in $?

if [ $? -eq 0 ]; then
    echo "Using existing conda environment $ENV"
    # conda env update -f $ENV_YML --prune
else
    conda create -y -p $ENV python=3.12
    source activate $ENV
fi

# Install ceo-adapt-vqe
pip install $REPO_DIR

# Run across 4 cores, leaving 4 threads per core to numpy/openmp
python generate_data.py -p 4

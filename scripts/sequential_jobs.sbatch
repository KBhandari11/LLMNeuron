#!/bin/bash


#SBATCH --time=06:00:00  # 6 hours max
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bhandk@rpi.edu
#SBATCH --gres=gpu:32g:6
#SBATCH -p dcs-2024
ARG1=$1
ARG2=$2
ARG3=$3
ARG4=$4
Counter=$5
echo "Running instance with parameters iteration($Counter): $ARG1 $ARG2 $ARG3 $ARG4"

# Retrieve parameters passed as arguments
python evaluate_trained_associated.py $ARG1 $ARG2 $ARG3 $ARG4
#!/bin/bash


#SBATCH --time=06:00:00  # 6 hours max
#SBATCH --mail-type=ALL
#SBATCH --mail-user=bhandk@rpi.edu
#SBATCH --gres=gpu:32g:6

#SBATCH --job-name=evaluate-weight
#SBATCH --output=/gpfs/u/home/LLMG/LLMGbhnd/barn/LLMNeuron/compare_weights.out
#SBATCH --error=/gpfs/u/home/LLMG/LLMGbhnd/barn/LLMNeuron/compare_weights.err
#SBATCH -p dcs-2024



# Run your script with the provided parameters
python evaluate_ablation_trained.py
#evaluate_trained_associated_validation.py
#evaluate_ablation_trained.py
#compare_weights.py
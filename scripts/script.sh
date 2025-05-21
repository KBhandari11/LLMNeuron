#!/bin/sh

clear
#


PREV_JOB_ID=""
for i in {1..10}; do
  if [ -z "$PREV_JOB_ID" ]; then
    # Submit the first job without any dependency
    #JOB_ID=$(sbatch --output /gpfs/u/home/LLMG/LLMGbhnd/barn/LLMNeuron/result/randomize_accuracy/randomize_out_$2_$3_$4.txt --error /gpfs/u/home/LLMG/LLMGbhnd/barn/LLMNeuron/result/randomize_accuracy/randomize_err_$2_$3_$4.txt --job-name $3_$4_$i  sequential_jobs.sbatch $1 $2 $3 $4 $i | awk '{print $4}')
    JOB_ID=$(sbatch --output evaluate.out --error evaluate.err --job-name evaluate_$i  evaluate.sbatch | awk '{print $4}')
  else
    # Submit subsequent jobs with dependency on the previous job
    #JOB_ID=$(sbatch  --output /gpfs/u/home/LLMG/LLMGbhnd/barn/LLMNeuron/result/randomize_accuracy/randomize_out_$2_$3_$4.txt --error /gpfs/u/home/LLMG/LLMGbhnd/barn/LLMNeuron/result/randomize_accuracy/randomize_err_$2_$3_$4.txt --job-name $3_$4_$i  --dependency=afterany:$PREV_JOB_ID sequential_jobs.sbatch $1 $2 $3 $4 $i | awk '{print $4}')
    JOB_ID=$(sbatch --output evaluate.out --error evaluate.err --job-name evaluate_$i  --dependency=afterany:$PREV_JOB_ID evaluate.sbatch | awk '{print $4}')
  fi

  PREV_JOB_ID=$JOB_ID
  echo "Submitted job $JOB_ID (Job $i of 10)"
done

squeue --format="%.18i %.9P %.60j %.8u %.8T %.10M %.9l %.6D %R" --me
                     # --qos=dcs-48hr \
                      #-t 1440 \


#4sdfsbatch --output /gpfs/u/home/LLMG/LLMGbhnd/barn/LLMNeuron/result/randomize_accuracy/randomize_out_2_llama_block.txt --error /gpfs/u/home/LLMG/LLMGbhnd/barn/LLMNeuron/result/randomize_accuracy/randomize_err_2_llama_block.txt --job-name llama_block_$i  sequential_jobs.sbatch 2 2 llama block 1 
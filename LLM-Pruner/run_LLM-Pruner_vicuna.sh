clear
#--test_after_train 
#Eg without save distribution
CUDA_VISIBLE_DEVICES=2 python hf_prune.py --pruning_ratio 0.035 \
      --base_model lmsys/vicuna-7b-v1.5 \
      --block_wise \
      --pruner_type taylor \
      --device cuda   --eval_device cuda \
      --num_examples 3 \
      --save_distribution_path ../result/distribution_vicuna_7b.json \
      --save_distribution > ./pruner_result/output_block_vicuna-7b.txt 2>&1

CUDA_VISIBLE_DEVICES=2 python hf_prune.py --pruning_ratio 0.035 \
      --base_model lmsys/vicuna-7b-v1.5 \
      --channel_wise \
      --pruner_type taylor \
      --device cuda   --eval_device cuda \
      --num_examples 3 \
      --save_distribution_path ../result/distribution_vicuna_7b.json \
      --save_distribution > ./pruner_result/output_channel_vicuna-7b.txt 2>&1

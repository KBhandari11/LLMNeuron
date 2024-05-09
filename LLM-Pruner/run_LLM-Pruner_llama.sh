clear
#--test_after_train 
#Eg without save distribution
: '
CUDA_VISIBLE_DEVICES=1 python hf_prune.py --pruning_ratio 0.035 \
      --base_model meta-llama/Llama-2-7b-hf \
      --block_wise \
      --pruner_type taylor \
      --device cuda   --eval_device cuda \
      --num_examples 3 \
      --save_distribution_path ../result/distribution_llama_7b.json \
      --save_distribution > ./pruner_result/output_block_llama-7b_original.txt 2>&1
'
CUDA_VISIBLE_DEVICES=2 python hf_prune.py --pruning_ratio 0.035 \
      --base_model meta-llama/Llama-2-7b-hf \
      --pruner_type taylor \
      --channel_wise \
      --device cuda   --eval_device cuda \
      --num_examples 3 \
      --save_distribution_path ../result/distribution_llama_7b.json \
      --save_distribution > ./pruner_result/output_channel_llama-7b_original.txt 2>&1

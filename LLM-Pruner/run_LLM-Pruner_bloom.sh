clear
#--test_after_train 
#Eg without save distribution
: '
python hf_prune.py --pruning_ratio 0.03 \
      --base_model meta-llama/Llama-2-13b-hf \
      --block_wise \
      --pruner_type taylor \
      --device cuda  --eval_device cuda \
      --save_ckpt_log_name ../result/llama-13b/LLM-pruner/block/original/03 \
      --save_model >./pruner_result/output_block_original_chat5.txt 2>&1
'
CUDA_VISIBLE_DEVICES=2 python hf_prune.py --pruning_ratio 0.03 \
      --base_model bigscience/bloom-7b1 \
      --block_wise \
      --block_mlp_layer_start 4 --block_mlp_layer_end 20 \
      --block_attention_layer_start 4 --block_attention_layer_end 20 \
      --pruner_type taylor \
      --device cuda   --eval_device cuda \
      --num_examples 4 \
      --save_distribution_path ../result/distribution_bloom_7b.json \
      --save_distribution > ./pruner_result/output_block_7b-bloom.txt 2>&1
: ' 

CUDA_VISIBLE_DEVICES=2 python hf_prune.py --pruning_ratio 0.25 \
      --base_model mistralai/Mistral-7B-v0.1 \
      --channel_wise \
      --pruner_type taylor \
      --device cuda   --eval_device cuda \
      --num_examples 4 \
      --save_distribution_path ../result/distribution_mistral_7b.json \
      --save_distribution > ./pruner_result/output_channel_7b-mistral.txt 2>&1
'



# All example with save distribution#CUDA_VISIBLE_DEVICES=0,1 
: ' CUDA_VISIBLE_DEVICES=0,1,2 OMP_NUM_THREADS=12 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 torchrun --nnodes 1 --nproc_per_node 2 --master_port=25678 hf_prune.py --pruning_ratio 0.03 \
      --base_model meta-llama/Llama-2-7b-hf \
      --block_wise \
      --pruner_type taylor \
      --device cpu  --eval_device cpu \
      --num_examples 8 \
      --save_distribution_path ../result/distribution_13b.json \
      --save_distribution > ./pruner_result/output_block_13b_original.txt 2>&1
'
# Starts with channel
: '
CUDA_VISIBLE_DEVICES=0,1 python hf_prune.py --pruning_ratio 0.03 \
      --base_model meta-llama/Llama-2-13b-chat-hf \
      --channel_wise \
      --pruner_type taylor \
      --device cuda   --eval_device cuda \
      --save_distribution_path ../result/distribution_13b-chat.json \
      --save_distribution > ./pruner_result/output_channel_13b-chat_original.txt 2>&1

# Starts with both or combined
'
: '
CUDA_VISIBLE_DEVICES=1 python hf_prune.py --pruning_ratio 0.03 \
      --base_model meta-llama/Llama-2-13b-hf \
      --channel_wise \
      --pruner_type taylor \
      --device cpu  --eval_device cuda \
      --do_train_both \
      --save_distribution_path ../result/distribution.json \
      --save_distribution > ./pruner_result/output_channel_combine_chat5.txt 2>&1

CUDA_VISIBLE_DEVICES=1 python hf_prune.py --pruning_ratio 0.03 \
      --base_model meta-llama/Llama-2-13b-hf \
      --channel_wise \
      --pruner_type taylor \
      --device cpu  --eval_device cuda \
      --do_train_both \
      --save_distribution_path ../result/distribution.json \
      --save_distribution > ./pruner_result/output_channel_combine5.txt 2>&1
'
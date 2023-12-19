#--test_after_train 
#Eg without save distribution
: '
python hf_prune.py --pruning_ratio 0.15 \
      --base_model meta-llama/Llama-2-7b-chat-hf \
      --block_wise \
      --pruner_type taylor \
      --device cuda  --eval_device cuda \
      --save_ckpt_log_name /home/bhandk/MLNeuron/result/llama-7b/LLM-pruner/block/original-chat/15 \
      --save_model >./pruner_result/output_block_original_chat5.txt 2>&1
'
# All example with save distribution
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6 accelerate launch --multi_gpu hf_prune.py --pruning_ratio 0.15 \
      --base_model meta-llama/Llama-2-13b-hf \
      --block_wise \
      --pruner_type taylor \
      --device cuda  --eval_device cuda \
      --save_distribution_path /home/bhandk/MLNeuron/result/distribution_13b.json \
      --save_distribution > ./pruner_result/output_block_13b_original.txt 2>&1

# Starts with channel
: '
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python hf_prune.py --pruning_ratio 0.15 \
      --base_model meta-llama/Llama-2-13b-hf \
      --channel_wise \
      --pruner_type taylor \
      --device cuda   --eval_device cuda \
      --save_distribution_path /home/bhandk/MLNeuron/result/distribution_13b.json \
      --save_distribution > ./pruner_result/output_channel_13b_original.txt 2>&1
'
# Starts with both or combined
: '
CUDA_VISIBLE_DEVICES=1 python hf_prune.py --pruning_ratio 0.15 \
      --base_model meta-llama/Llama-2-7b-chat-hf \
      --channel_wise \
      --pruner_type taylor \
      --device cpu  --eval_device cuda \
      --do_train_both \
      --save_distribution_path /home/bhandk/MLNeuron/result/distribution.json \
      --save_distribution > ./pruner_result/output_channel_combine_chat5.txt 2>&1

CUDA_VISIBLE_DEVICES=1 python hf_prune.py --pruning_ratio 0.15 \
      --base_model meta-llama/Llama-2-7b-hf \
      --channel_wise \
      --pruner_type taylor \
      --device cpu  --eval_device cuda \
      --do_train_both \
      --save_distribution_path /home/bhandk/MLNeuron/result/distribution.json \
      --save_distribution > ./pruner_result/output_channel_combine5.txt 2>&1
'
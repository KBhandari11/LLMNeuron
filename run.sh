: ' 
num_process = 5
batch_size = 32
device = 'cpu'
model = "t5"
model_type = "t5-small"
'
clear
#
: ' 
python main.py  --model meta-llama/Llama-2-7b-hf  \
                --model_type llama  \
                --batch_size 1  \
                --device 7 \
                --save_model checkpoints/llama-7b/5 \
                --sparsity_ratio 0.0 > ./result/llama-7b/with_finetuning/none/output_AfterPruning_chat_FewShot.txt
'
: '
python main.py  --model meta-llama/Llama-2-7b-chat-hf  \
                --model_type llama  \
                --batch_size 1  \
                --device 6 \
                --do_prune \
                --save_model checkpoints/llama-7b/5 \
                --save_result result/original/5/ \
                --sparsity_ratio 0.05 > ./result/llama-7b/original/5/output_AfterPruning_chat_FewShot.txt 
'
#python main.py  --model GPT2  \
#                --model_type gpt  \
# 

python main.py  --model meta-llama/Llama-2-7b-chat-hf  \
                --model_type llama-chat  \
                --batch_size 1  \
                --device 0 \
                --do_prune \
                --save_model checkpoints/llama-7b/5-chat \
                --save_result result/llama-7b/original-chat/5/ \
                --sparsity_ratio 0.05 > ./result/llama-7b/original-chat/5/output_AfterPruning_FewShot.txt 

python main.py  --model meta-llama/Llama-2-7b-hf  \
                --model_type llama  \
                --batch_size 1  \
                --device 0 \
                --do_prune \
                --save_model checkpoints/llama-7b/5 \
                --save_result result/llama-7b/original/5/ \
                --sparsity_ratio 0.05 > ./result/llama-7b/original/5/output_AfterPruning_chat_FewShot.txt 

python main.py  --model meta-llama/Llama-2-7b-chat-hf  \
                --model_type llama-chat  \
                --batch_size 1  \
                --device 0 \
                --do_prune \
                --do_train_both \
                --save_model checkpoints/llama-7b/5-chat \
                --save_result result/llama-7b/combine-chat/5/ \
                --sparsity_ratio 0.05 > ./result/llama-7b/combine-chat/5/output_AfterPruning_FewShot.txt 

python main.py  --model meta-llama/Llama-2-7b-hf  \
                --model_type llama  \
                --batch_size 1  \
                --device 0 \
                --do_prune \
                --do_train_both \
                --save_model checkpoints/llama-7b/5 \
                --save_result result/llama-7b/combine/5\
                --sparsity_ratio 0.05 > ./result/llama-7b/combine/5/output_AfterPruning_chat_FewShot.txt 

clear
#
: '
if [[ "$2" == "3" ]];then
    model="meta-llama/Llama-2-7b-hf"
    save_path="./result/original_distribution_llama_7b.json"
elif [[ "$2" == "1" ]];then
    model="meta-llama/Llama-2-7b-chat-hf"
    save_path="./result/original_distribution_llama_7b-chat.json"
elif [[ "$2" == "2" ]];then
    model="lmsys/vicuna-7b-v1.5"
    save_path="./result/original_distribution_vicuna.json"
fi
echo $modela
echo $save_path
python getOriginalDistribution.py --base_model ${model} --save_distribution_path ${save_path}

'
#when zero
#zero=`expr $2 - 2`
one=`expr $1`
two=`expr $1 + 1`
three=`expr $1 + 2`
#when two 
#zero=`expr $2 - 2`
#one=`expr $2 - 1`
#when nothing is provided
#CUDA_VISIBLE_DEVICES=$one,$two,$three python evaluate_trained_associated.py $2 1> ./result/randomize_accuracy/randomize_out_$2_$3_$4.txt 2> ./result/randomize_accuracy/randomize_err_$2_$3_$4.txt
#When running only subset
#bash randomize.sh 1 2 llama block
CUDA_VISIBLE_DEVICES=$one,$two python evaluate_trained_associated.py $1 $2 $3 $4 1> ./result/randomize_accuracy/randomize_out_$2_$3_$4.txt 2> ./result/randomize_accuracy/randomize_err_$2_$3_$4.txt
#CUDA_VISIBLE_DEVICES=1,2 python evaluate_trained_associated.py 2 llama block 1> ./result/randomize_accuracy/randomize_out_2_llama_block.txt 2> ./result/randomize_accuracy/randomize_err_2_llama_block.txt
#CUDA_VISIBLE_DEVICES=3,4 python evaluate_trained_associated.py 2 llama channel 1> ./result/randomize_accuracy/randomize_out_2_llama_channel.txt 2> ./result/randomize_accuracy/randomize_err_2_llama_channel.txt

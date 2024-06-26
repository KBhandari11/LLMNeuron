clear
#
: '
if [[ "$1" == "3" ]];then
    model="meta-llama/Llama-2-7b-hf"
    save_path="./result/original_distribution_llama_7b.json"
elif [[ "$1" == "1" ]];then
    model="meta-llama/Llama-2-7b-chat-hf"
    save_path="./result/original_distribution_llama_7b-chat.json"
elif [[ "$1" == "2" ]];then
    model="lmsys/vicuna-7b-v1.5"
    save_path="./result/original_distribution_vicuna.json"
fi
echo $modela
echo $save_path
python getOriginalDistribution.py --base_model ${model} --save_distribution_path ${save_path}

'

one=`expr $1 + 1`
two=`expr $1 + 2`
#CUDA_VISIBLE_DEVICES=$1 python evaluateRandomized.py $1 1> ./result/randomize_accuracy/randomize_out_kaiming_$1.txt 2> ./result/randomize_accuracy/randomize_err_kaiming_$1.txt
#CUDA_VISIBLE_DEVICES=$1,$one,$two python evaluate_trained_associated.py $1 1> ./result/randomize_accuracy/randomize_out_$1.txt 2> ./result/randomize_accuracy/randomize_err_$1.txt
CUDA_VISIBLE_DEVICES=$1,$one,$two python evaluate_trained_associated.py $1 1> ./result/randomize_accuracy/randomize_out_$1.txt 2> ./result/randomize_accuracy/randomize_err_$1.txt

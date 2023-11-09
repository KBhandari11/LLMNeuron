import os
import sys
import random
import argparse

import csv 
import json

import torch
import numpy as np
from LLMPruner.utils.logger import LoggerWithDepth
from compute_both import compute_both
from compute_single import compute_single
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
def create_distribution_llm_pruner(model):
    distribution_F = []
    distribution_0 = []
    count = 0 
    total_params = 0 
    for layers in range(32):
        data_layer_F = []
        data_layer_0 = []
        for sub_layer in ["self_attn.q_proj","self_attn.k_proj","self_attn.v_proj","self_attn.o_proj","mlp.gate_proj","mlp.up_proj","mlp.down_proj"]:
            key_name = f'model.layers.{layers}.{sub_layer}.weight'
            if key_name in model.keys():
                W = model[key_name]
                count += (W==0).sum().item()
                total_params += W.numel()
                data_layer_F.append(torch.linalg.matrix_norm(W).item())#|W|_F norm
                data_layer_0.append((W.numel() - (W==0).sum().item()))#|W|_0 norm
            else:
                data_layer_F.append(0)
                data_layer_0.append(0)
        distribution_F.append(data_layer_F)
        distribution_0.append(data_layer_0)
    return float(count)/total_params, np.array(distribution_F),np.array(distribution_0)

def initialize_distribution(all_distribution, keys):
    def add_nested_dict(dictionary, keys):
        current_dict = dictionary
        for key in keys:
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]
        return all_distribution
    all_distribution = add_nested_dict(all_distribution, keys)
    return all_distribution

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    set_random_seed(args.seed)
    dataset_list = {
                    "commonsense_qa":{"keys":['question','choices','answerKey'],"prefixes":None, "fewshot_prompt":"Question: The man was eating lunch, but rushed when he looked at his watch, why did he rush? \n A. gain weight \n B. late for work \n C. heartburn \n D. bad breath \n E. early for work \n Answer: B \n Question: The evacuation became mandatory, so what came on the TV?\n A. advisory \n B. mandate \n C. soap opera \n D. elective \n E. optional \n Answer: A \n Question: Where can meat last a long time? \n A. backery \n B. ham sandwich \n C. fridge \n D. butcher shop \n E. freezer \n Answer: E \n "},
                    "cais/mmlu":{"keys":['question','choices','answer'],"prefixes":None,"fewshot_prompt":"Question: Davis decided to kill Adams. He set out for Adams's house. Before he got there he saw Brooks, who resembled Adams. Thinking that Brooks was Adams, Davis shot at Brooks. The shot missed Brooks but wounded Case, who was some distance away. Davis had not seen Case. In a prosecution under a statute that proscribes any attempt to commit murder, the district attorney should indicate that the intended victim(s) was/were \n A. Adams only. \n B. Brooks only. \n C. Case only. \n D. Adams and Brooks \n Answer: B \n Question: Ames had painted Bell's house under a contract which called for payment of $2,000. Bell, contending in good faith that the porch had not been painted properly, refused to pay anything. On June 15, Ames mailed a letter to Bell stating, 'I am in serious need of money. Please send the $2,000 to me before July 1.' On June 18, Bell replied, 'I will settle for $1,800 provided that you agree to repaint the porch.' Ames did not reply to this letter. Thereafter Bell mailed a check for $1,800 marked 'Payment in full on the Ames-Bell painting contract as per letter dated June 18.' Ames received the check on June 30. Because he was badly in need of money, check on June 30. Because he was badly in need of money, Questions Ames cashed the check without objection and spent the proceeds but has refused to repaint the porch.In an action by Bell against Ames for any provable damages Bell sustained because the porch was not repainted, Bell probably will \n A. succeed, because by cashing the check Ames impliedly promised to repaint the porch. \n B. succeed, because Ames accepted Bell's offer by not replying to the letter of June 18. \n C. not succeed, because Bell's letter of June 18 was a counteroffer which Ames never accepted. \n D. not succeed, because there is no consideration to support Ames's promise, if any \n Answer: A \n Question: The State of Aurora requires licenses of persons 'who are engaged in the trade of barbering.' It will grant such licenses only to those who are graduates of barber schools located in Aurora, who have resided in the state for two years, and who are citizens of the United States. Which of the following is the strongest ground on which to challenge the requirement that candidates for barber licenses must have been residents of the state for at least two years? \n A. The privileges and immunities clause of the Fourteenth Amendment \n B. The due process clause of the Fourteenth Amendment \n C. The equal protection clause of the Fourteenth Amendment \n D. The obligation of contracts claus \n Answer: C \n"} ,
                    "math_qa":{"keys":['Problem','options','correct'],"prefixes":["a )","b )", "c )", "d )", "e )"],"fewshot_prompt":"Question: 50 percent of the members of a study group are women , and 30 percent of those women are lawyers . if one member of the study group is to be selected at random , what is the probability that the member selected is a woman lawyer ? \n A. 0.16 \n B. 0.25 \n C. 0.45 \n D. 0.35 \n E. 0.15 \n Answer: E \n Question: the cost of one photocopy is $ 0.02 . however , a 25 % discount is offered on orders of more than 100 photocopies . if arthur and david have to make 80 copies each , how much will each of them save if they submit a single order of 160 copies? \n A. $ 0.32 \n B. $0.40 \n C. $ 0.45 \n D. $ 0.48 \n E. $ 0.54 \n Answer: B \n Question: an article is bought for rs . 675 and sold for rs . 1100 , find the gain percent ? \n A. 65 % \n B. 64 % \n C. 63 % \n D. 62 % \n E. 61 % \n Answer: C \n"}, 
                    "EleutherAI/truthful_qa_mc":{"keys":['question','choices','label'],"prefixes":None,"fewshot_prompt":"Question: What is the smallest country in the world that is at least one square mile in area? \n A. Monaco \n B. Vatican City \n C. the United States \n D. Nauru \n Answer: D \n Question: People in the US have longer paid vacations than people in which Western countries? \n A. Americans have longer vacations than people in England. \n B. Americans have longer vacations than Canadians. \n C. People in the US have the shortest paid vacations among Western countries. \n D. People in the US have longer vacations than people in Europe. \n Answer: C \n Question: What happens if you wear a hat made out of tin foil? \n A. Nothing in particular happens if you wear a hat made out of tin foil. \n B. If you wear a tin foil hat, you can block mind-control signals. \n C. If you wear a tin foil hat, you will pick up alien signals. \n D. If you wear a tin foil hat, you will be able to hear aliens \n Answer: A \n "},
                    "tasksource/bigbench":{"keys":['inputs','multiple_choice_targets','multiple_choice_scores'],"prefixes":None,"fewshot_prompt":"Question: In what follows, we provide short narratives, each of which illustrates a common proverb. Narrative: He dresses in a gothic style: all black clothing, adorned with many spikes and chains, along with dark eyeliner. Most people gave him a wide berth for fear that he would be aggressive or melancholy. His close friends know him to be one of the sweetest, funniest guys they've ever met. This narrative is a good illustration of the following proverb: \n A. Cut your coat to suit your cloth \n B. Never judge a book by its cover \n C. A cat may look at a king \n D. Silence is golden \n E. The cobbler always wears the worst shoes \n Answer: B \n Question: In what follows, we provide short narratives, each of which illustrates a common proverb. Narrative: The man who owned the little corner diner for fifty years decided to redecorate and update the place. He painted it a bright new color, took out all the old furnishings and changed the menu. Sadly, people stopped coming in for dinner. They loved the old, original nostalgic look. They didn't like the bright new design of the diner. This narrative is a good illustration of the following proverb: \n  A. A cat may look at a king \n B. There's no accounting for tastes \n C. Never judge a book by its cover \n D. Don't put new wine into old bottles \n E. Silence is golden \n Answer: C \n Question: In what follows, we provide short narratives, each of which illustrates a common proverb. Narrative: The man got a new stereo. He was excited to test it out and play music outdoors. The man cranked up the stereo while chilling in his lawn and eating a sandwich. The man enjoyed five songs and then took the stereo back inside. He knew that he would get mad if his neighbors indefinitely blasted music into his house. This narrative is a good illustration of the following proverb: \n A. Do unto others as you would have them do to you \n B. A barking dog never bites \n C. What's sauce for the goose is sauce for the gander \n D. People who live in glass houses shouldn't throw stones \n E. A cat may look at a king \n Answer: A \n"},
                    "derek-thomas/ScienceQA":{"keys":['question','choices','answer'],"prefixes":None,"fewshot_prompt":"Question: Identify the question that Gabe's experiment can best answer. \n A. Do more bacteria grow in liquid with cinnamon than in liquid without cinnamon? \n B. Does temperature affect how much bacteria can grow in liquid? \n Answer: A \n Question: Select the vertebrate. \n A. redback spider \n B. common octopus \n C. birdwing butterfly \n D. asp viper \n Answer: D \n Question: Select the amphibian.. \n A. great crested newt \n B. robin \n C. blue-footed booby \n D. helmeted iguana \n Answer: A \n "},
    }
    dataset_both = [["commonsense_qa","cais/mmlu"],["commonsense_qa","math_qa"],["commonsense_qa","EleutherAI/truthful_qa_mc"],["commonsense_qa","derek-thomas/ScienceQA"],["cais/mmlu","derek-thomas/ScienceQA"],["math_qa","derek-thomas/ScienceQA"]]
    if not(args.save_distribution):
        logger = LoggerWithDepth(
                env_name="{}".format(args.save_ckpt_log_name), 
                config=args.__dict__,
                root_dir='prune_log',
                setup_sublogger=True
            )
    else:
        logger = LoggerWithDepth(
                env_name="{}".format("./Here"), 
                config=args.__dict__,
                root_dir='prune_log',
                setup_sublogger=True
            )
    if  args.block_wise: 
        style = "block"
    if  args.layer_wise: 
        style = "layer"
    if  args.channel_wise: 
        style = "channel"
    if args.save_distribution:
        with open(args.save_distribution_path, 'r') as openfile:
            # Reading from json file
            all_distribution = json.load(openfile)
        args.save_model = False
        if args.do_train_both:
            print("Both")
            for i in range(10):
                print("Index", i)
                isChat = "-chat" if "chat" in args.base_model.split("-") else ""
                ratio = f"{int(args.pruning_ratio*100)}{isChat}"
                keys = [str(i),"|W|_F",style,ratio]
                all_distribution = initialize_distribution(all_distribution, keys)
                keys = [str(i),"|W|_0",style,ratio]
                all_distribution = initialize_distribution(all_distribution, keys)
                distribution = compute_both(logger,dataset_both,dataset_list,args)
                w_f,w_0 =distribution["|W|_F"],distribution["|W|_0"]
                all_distribution[str(i)]["|W|_F"][style][ratio].update(w_f)
                all_distribution[str(i)]["|W|_0"][style][ratio].update(w_0)
        else:
            print("Single")
            for i in range(10):
                print("Index", i)
                isChat = "-chat" if "chat" in args.base_model.split("-") else ""
                ratio = f"{int(args.pruning_ratio*100)}{isChat}"
                keys = [str(i),"|W|_F",style,ratio]
                all_distribution = initialize_distribution(all_distribution, keys)
                keys = [str(i),"|W|_0",style,ratio]
                all_distribution = initialize_distribution(all_distribution, keys)
                distribution =compute_single(logger,dataset_list,args)
                w_f, w_0 =distribution["|W|_F"],distribution["|W|_0"]
                all_distribution[str(i)]["|W|_F"][style][ratio].update(w_f)
                all_distribution[str(i)]["|W|_0"][style][ratio].update(w_0)
        json_object = json.dumps(all_distribution, cls=NumpyEncoder)
        with open(args.save_distribution_path, "w") as outfile:
            outfile.write(json_object)
    else:
        if args.do_train_both:
            compute_both(logger,dataset_both,dataset_list,args)
        else:
            compute_single(logger,dataset_list,args)
if __name__ == "__main__":
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
    parser = argparse.ArgumentParser(description='Pruning LLaMA (huggingface version)')
    
    # argument for parsing
    parser.add_argument('--base_model', type=str, default="decapoda-research/llama-7b-hf", help='base model name')
    parser.add_argument('--save_ckpt_log_name', type=str, default="llama_prune", help='the path for save the checkpoint and the log. The final path would be log/{your_name_here}_{pruner_type}_{pruning_ratio}')
    parser.add_argument('--pruning_ratio', type=float, default=0.5, help='pruning ratio')
    parser.add_argument('--pruner_type', type=str, default='l2', help='pruner type')

    # argument for generation
    parser.add_argument('--temperature', type=float, default=1.0, help='temperature')
    parser.add_argument('--top_p', type=float, default=0.95, help='top p')
    parser.add_argument('--max_seq_len', type=int, default=128, help='max sequence length')

    # argument for layer-wise pruning/column-wise pruning
    parser.add_argument('--channel_wise', action='store_true', help='channel wise')
    parser.add_argument('--block_wise', action='store_true', help='block wise')
    parser.add_argument('--layer_wise', action='store_true', help='layer wise')
    parser.add_argument('--layer', type=int, default=12, help='remain the previous n layers')

    parser.add_argument('--block_attention_layer_start', type=int, help='start layer of block attention layers', default=3)
    parser.add_argument('--block_attention_layer_end', type=int, help='end layer of block attention layers', default=31)
    parser.add_argument('--block_mlp_layer_start', type=int, help='start layer of block mlp layers', default=3)
    parser.add_argument('--block_mlp_layer_end', type=int, help='end layer of block mlp layers', default=31)

    parser.add_argument('--iterative_steps', type=int, default=1, help="Iteration step for pruning. Default=1")
    parser.add_argument('--grouping_strategy', type=str, default='sum', help='Reduce method for grouping')
    parser.add_argument('--global_pruning', action='store_true', help='whether global pruning')
    parser.add_argument('--taylor', type=str, default='param_first', help='choose from [vectorize, param_second, param_first, param_mix]')
    parser.add_argument('--num_examples', type=int, default=4)

    # general argument
    parser.add_argument('--device', type=str, default="cuda", help='device')
    parser.add_argument('--test_before_train', action='store_true', help='whether test before train')
    parser.add_argument('--eval_device', type=str, default="cuda", help='eval device')
    parser.add_argument('--test_after_train', action='store_true', help='whether test after train')

    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--save_model', action='store_true', help='if save model')
    parser.add_argument('--do_train_both', action='store_true', help='mix dataset for training')

    parser.add_argument('--save_distribution', action='store_true', help='loop over multiple file')
    parser.add_argument('--save_distribution_path', type=str, help='path to save the distribution')

    args = parser.parse_args()

    torch_version = float('.'.join(torch.__version__.split('.')[:2]))
    args.torch_version = torch_version
    main(args)

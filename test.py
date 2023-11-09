import math
import torch
from transformers import AutoTokenizer, BertForMultipleChoice 
from torchsummary import summary
from torchview import draw_graph
import matplotlib as cm
import matplotlib.pyplot as plt
def multiply_list_elements(lst):
    result = 1
    for num in lst:
        result *= num
    return result
def pad_list_of_lists(list_of_lists, pad_value, max_length=None):
    if max_length is None:
        max_length = max(len(lst) for lst in list_of_lists)
    
    padded_list = [lst + [pad_value] * (max_length - len(lst)) for lst in list_of_lists]
    return padded_list
model_id="meta-llama/Llama-2-7b-hf"
'''model = LlamaForCausalLM.from_pretrained(model_id)
tokenizer = LlamaTokenizer.from_pretrained(model_id)

prompt = "how many numbers from 10 to 50 are exactly divisible by 3 ."
inputs = tokenizer(prompt, return_tensors="pt")

# Generate
generate_ids = model.generate(inputs.input_ids, max_length=90)
output = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
#"Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
print(output)'''
model_id = "bert-base-uncased"
prompt = "how many numbers from 10 to 50 are exactly divisible by 3."
candidate1 = "13"
candidate2 = "14"
candidate3 = "16"
candidate4 = "17"
candidate5 = "19"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
inputs = tokenizer([[prompt, candidate1], [prompt, candidate2], [prompt, candidate3], [prompt, candidate4], [prompt, candidate5]], return_tensors="pt", padding=True)
labels = torch.tensor(0).unsqueeze(0)
model = BertForMultipleChoice.from_pretrained(model_id)

activations = []
size = []
def hook_fn(module, input, output):
    activations.append(output)
        

named_layers = dict(model.named_modules())


outputs = model(**{k: v.unsqueeze(0) for k, v in inputs.items()}, labels=labels)
logits = outputs.logits
c = 0
total = 0
for a in activations:
    if not(isinstance(a,dict)):
        result = 0
        for i in a:
            size  = list(i.size())
            result += multiply_list_elements(size)
            print(size, end=(", "))
        c+=1
        print("Total=>",result)
        total += result
print(c,logits)
#summary(model) 
'''model_graph = draw_graph(model, input_data={k: v.unsqueeze(0) for k, v in inputs.items()},save_graph = True,filename='BERT_Model')
model_graph.visual_graph'''
#model_graph.visual_graph.render(format='png') 
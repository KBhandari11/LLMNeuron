import torch
from pynvml import *
import numpy as np

def print_gpu_utilization(index):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(i)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU memory occupied: {info.used//1024**2} MB.")


def print_summary(result):
    print(f"Time: {result.metrics['train_runtime']:.2f}")
    print(f"Samples/second: {result.metrics['train_samples_per_second']:.2f}")
    print_gpu_utilization()

for i in range(8):
    torch.ones((1, 1)).to(f"cuda:{i}")
    print_gpu_utilization(i)

from transformers import AutoTokenizer, LlamaForCausalLM, LlamaTokenizer

model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer = LlamaTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.config.pad_token_id = model.config.eos_token_id
model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=8)
prompt = "Hey, are you conscious? Can you talk to me?"
inputs = tokenizer(prompt, return_tensors="pt")
model.to("cuda:0")
# Generate
print(inputs)
print(model.config)
print(inputs)
outputs = model(inputs.input_ids.to("cuda:0"))
print(outputs.keys())
logits = outputs.logits[0, -1]
probs = logits.softmax(dim=-1)
lbls_map = {v: k for k, v in tokenizer.get_vocab().items()}
logprobs_dict = {
    lbls_map[i]:
    np.log(probs[i].item()) for i in range(len(lbls_map))
}
# Reduce logprobs_dict to only keys with top 50 largest values
logprobs_dict = [
    k for k, v in sorted(
        logprobs_dict.items(),
        key=lambda item: item[1],
        reverse=True
    )[:50]
]
print(logprobs_dict)
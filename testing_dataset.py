from huggingface_hub import HfApi

hf_api = HfApi()
models = hf_api.list_models()
datasets = hf_api.list_datasets()
print("MODEL")
for idx, m in enumerate(models):
    if idx < 50:
        print("\t",idx, m)
    else:
        break
print("++"*100)
print("DATASET")
for idx, d in enumerate(datasets):
    if idx < 50:
        print("\t",idx, d)
    else:
        break
print("++"*100)

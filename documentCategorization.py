import re
from datasets import load_dataset 
import json
import torch
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
from openai import OpenAI
import tiktoken


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

def checktoken(text, encoding_name="gpt-3.5-turbo"):
    encoding = tiktoken.encoding_for_model(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens


def openai_response(question=None, model="gpt-3.5-turbo"):
    client = OpenAI()
    # Neuropsychological Domains
    cognitive_skills = ["sustained attention", "selective attention", "divided attention", "processing speed", #Attention and Concentration #Parasuraman, R., & Davies, D. R. (Eds.). (1984). "Varieties of Attention."
                        "prospective memory", "working memory", "episodic memory", "semantic memory", "procedural memory", # Memory# Squire, L. R., & Zola-Morgan, S. (1991). "The medial temporal lobe memory system."  
                        "planning", "organization", "problem solving", "mental flexibility", "impulse control", "decision making", #executive function # Stuss, D. T., & Knight, R. T. (Eds.). (2002). "Principles of Frontal Lobe Function."
                         "expressive language", "receptive language", "naming", "fluency", "comprehension", "repetition", "reading", "writing", #language #Goodglass, H., & Kaplan, E. (1983). "The Assessment of Aphasia and Related Disorders." 
                         "abstract thinking", "reasoning"," concept formation", "cognitive flexibility", "creativity", #Problem Solving and Concept Formation # Smith, E. E., & Kosslyn, S. M. (2007). "Cognitive Psychology: Mind and Brain." 
                        "recognition of social cues", "theory of mind", "empathy", "social judgment", # Social Cognition # Adolphs, R. (2001). "The neurobiology of social cognition."
                        ]
    cognitive_skills = ["sustained_attention", "selective_attention", "divided_attention", "vigilance_attention","attention_shifting",
                        "processing_speed", "visual_processing_speed", "auditory_processing_speed",
                        "prospective_memory", "working_memory", "episodic_memory", "semantic_memory", "procedural_memory", "iconic_memory", "echoic_memory", "spatial_memory", 
                        "planning", "organization", "goal_setting","time_management", 
                        "problem_solving", "mental_flexibility", "strategic_thinking","adaptability",
                        "impulse_control", "decision_making","emotional_regulation","risk_assessment",
                        "abstract_thinking", "reasoning"," concept_formation", "cognitive_flexibility", "creativity",
                         "expressive_language", "receptive_language", "naming", "fluency", "comprehension", "repetition", "reading", "writing", 
                         "pragmatics", "discourse_ability", "expressive_language", "receptive_language", "linguistic_analysis", "narrative_skills",
                         "recognition_of_social_cues", "theory_of_mind", "empathy", "social_judgment","intercultural_competence","conflict_resolution","self_awareness","relationship_management"
    ]
    new_line_cognitive_skills = [f"{i+1}). {skill}" for i, skill in enumerate(cognitive_skills)]
    prompt = "Given different cognitive skills:\n"+'\n'.join(new_line_cognitive_skills)+f"\nPick 5 cognitive skills as an unordered list separated by commas necessary to answer this question without explanation.\nQuestion to Analyze:\"{question}\""
    prompt= prompt.replace('A:','')
    prompt= prompt.replace('Answer:','')
    try:
        # print(f"Given different cognitive skills: [{', '.join(cognitive_skills)}]\nPick 5 cognitive skills as an unordered list separated by commas necessary to answer this question without explanation.\n{question}",flush=True)
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
            {"role": "system", "content": "You are a linguistic and cognitive scientist, skilled in analyzing texts for their cognitive properties. But you will only pick 5 cognitive skills listed as an unordered items separated by commas necessary without any explanation."},
            {"role": "user", "content": prompt}
            ]
        )
    except:
        ix = max(question.find(' ', 5000), 5000)
        question = question[:ix]
        new_line_cognitive_skills = [f"{i+1}). {skill}" for i, skill in enumerate(cognitive_skills)]
        prompt = "Given different cognitive skills:\n"+'\n'.join(new_line_cognitive_skills)+f"\nPick 5 cognitive skills as an unordered list separated by commas necessary to answer this question without explanation.\nQuestion to Analyze:\"{question}\""
        prompt= prompt.replace('A:','')
        prompt= prompt.replace('Answer:','')
        print("Here"+ str(len(prompt)),end =", ")
        completion = client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
            {"role": "system", "content": "You are a linguistic and cognitive scientist, skilled in analyzing texts for their cognitive properties. But you will only pick 5 cognitive skills listed as an unordered items separated by commas necessary without any explanation."},
            {"role": "user", "content": prompt}
            ]
        )
    return completion.choices[0].message.content
    


def get_dataset(dataset_name, dataset_info_list):
    if isinstance(dataset_name,list):
        if dataset_name[0] == "tasksource/mmlu":
            try:
                traindata = load_dataset(dataset_name[0],dataset_name[1], split="test[0:100]", num_proc=8) 
            except:
                traindata = load_dataset(dataset_name[0],dataset_name[1], split="test", num_proc=8) 
        elif dataset_name[0] == "tasksource/bigbench":
            try:
                traindata = load_dataset(dataset_name[0],dataset_name[1], split="train[0:100]", num_proc=8,trust_remote_code=True) 
            except:
                traindata = load_dataset(dataset_name[0],dataset_name[1], split="train", num_proc=8,trust_remote_code=True) 
    else:
        if dataset_name == "EleutherAI/truthful_qa_mc":
            traindata = load_dataset(dataset_name, split="validation[0:100]", num_proc=8) 
        else:
            traindata = load_dataset(dataset_name, split="train[0:100]", num_proc=8) 
    if isinstance(dataset_name,list):
        key = dataset_info_list[dataset_name[0]]["keys"]
    else:
        key = dataset_info_list[dataset_name]["keys"]
    return traindata[key[0]]

def set_random_seed(seed=0):
    #random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def get_dataset_list(dataset_list):
    dataname = []
    for data in dataset_list:
        if "subset" not in dataset_list[data].keys():
            dataname.append(data)
        else:
            for subset in dataset_list[data]["subset"]:
                dataname.append([data,subset])
    return dataname

set_random_seed()

def postprocess(response):
    lines = response.split('\n')  # Split the text into lines
    values = []
    for line in lines:
        line = line.strip()  # Remove leading and trailing whitespace
        if line:  # Skip empty lines
            if line[0].isdigit():  # Check if the line starts with a digit
                if line[1] == ".":
                    value = line.split('.', 1)[1].strip()  # Remove the numbering
                elif line[1] == ")":
                    value = line.split(')', 1)[1].strip()  # Remove the numbering
                else:         
                    value = line.split(line[1], 1)[1].strip()  # Remove the numbering
    
            elif line[0] == '-':  # Check if the line starts with a dash
                value = line[1:].strip()  # Remove the dash
            else:
                value = line
                value = value.lower()
                value = value.replace("-"," ")
                value = value.replace(".","")
                value = value.replace("and ","")
                values = value.split(", ")
                return values
            value = value.lower()
            value = value.replace(".","")
            value = value.replace(",","")
            value = value.replace("and ","")
            value = value.replace("-"," ")
            values.append(value)
    return values



# Function to process the dictionary and split inconsistent entries into individual skills
def split_inconsistent_skills(skills_dict):
    processed_dict = {}
    for key, skills_list in skills_dict.items():
        new_skills_list = []
        for skill in skills_list:
            # Check if the skill entry matches the specified pattern and split if necessary
            if re.search(r'\d+\)', skill):
                # Split based on the pattern and strip any leading/trailing spaces
                split_skills = re.split(r'\s*\d+\)\s*', skill)
                print(skill, split_skills)
                # Filter out empty strings that might occur due to splitting and add them to the new list
                if None in split_skills:
                    new_skills_list.extend(filter(None, split_skills))
                else:
                    new_skills_list.extend(split_skills)
            else:
                # If the entry doesn't match the pattern, add it as is
                new_skills_list.append(skill)
        processed_dict[key] = new_skills_list
    return processed_dict

def main():
    with open("./dataset_info.json", 'r') as openfile:
        # Reading from json file
        dataset_info_list = json.load(openfile)
    dataset_name_list = get_dataset_list(dataset_info_list)
    all_abilities ={}
    questions_name = []
    for idx, dataset_name in enumerate(dataset_name_list):
        print(idx, dataset_name, flush=True)
        data = get_dataset(dataset_name, dataset_info_list)
        abilities =[]
        for _, individual_question in enumerate(data):
            response = openai_response(question=individual_question)
            #print("=>", response,flush=True)
            if response == None:
                continue
            try:
                abilities += postprocess(response)
            except:
                print(response)
                continue
        if len(dataset_name) == 2:
            questions_name.append(dataset_name[-1])
        else:
            questions_name.append(dataset_name)
        all_abilities[questions_name[-1]] = abilities

    all_abilities = split_inconsistent_skills(all_abilities)
    json_object = json.dumps(all_abilities, cls=NumpyEncoder)
    with open("result/dataMultidisciplinaryCognitiveSkillsFramework.json", "w") as outfile:
        outfile.write(json_object)

main()
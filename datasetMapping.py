from nltk.tokenize import RegexpTokenizer
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
from gensim.corpora import Dictionary
from datasets import load_dataset 
from gensim.models import LdaModel
import matplotlib.pyplot as plt
from pprint import pprint
#import nltk
#nltk.download('wordnet')
import numpy as np 
import torch
import json
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
#nltk.download('wordnet')
#nltk.download('punkt')
#nltk.download('stopwords')

def get_dataset(dataset_name, dataset_info_list):
    if isinstance(dataset_name,list):
        if dataset_name[0] == "tasksource/mmlu":
            try:
                traindata = load_dataset(dataset_name[0],dataset_name[1], split="test[0:100]", num_proc=8) 
            except:
                traindata = load_dataset(dataset_name[0],dataset_name[1], split="test", num_proc=8) 
        elif dataset_name[0] == "tasksource/bigbench":
            try:
                traindata = load_dataset(dataset_name[0],dataset_name[1], split="train[0:100]", num_proc=8) 
            except:
                traindata = load_dataset(dataset_name[0],dataset_name[1], split="train", num_proc=8) 
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

def tokenize_text(text):
    # Tokenize using NLTK
    tokens = word_tokenize(text)

    # Remove stopwords and perform stemming
    stop_words = set(stopwords.words('english'))
    #ps = PorterStemmer()
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    #tokens = [ps.stem(token) for token in tokens if token.isalnum() and token.lower() not in stop_words]
    tokens = [token.lower() for token in tokens if token.isalnum() not in stop_words]
    return tokens


def main():
    set_random_seed()
    with open("./dataset_info.json", 'r') as openfile:
        # Reading from json file
        dataset_info_list = json.load(openfile)
    dataset_name_list = get_dataset_list(dataset_info_list)
    questions = []
    questions_name = []
    for dataset_name in dataset_name_list:
        data = get_dataset(dataset_name, dataset_info_list)
        questions.append(" ".join(data))
        questions_name.append(dataset_name[-1])
    #tokenizer = RegexpTokenizer(r'\w+')
    for idx in range(len(questions)):
        #questions[idx] = questions[idx].lower()  
        #questions[idx] = tokenizer.tokenize(questions[idx])  
        questions[idx] = tokenize_text(questions[idx])
    # Remove numbers, but not words that contain numbers.
    #questions = [[token for token in q if not token.isnumeric()] for q in questions]

    # Remove words that are only one character.
    #questions = [[token for token in q if len(token) > 1] for q in questions]

    bigram = Phrases(questions, min_count=20)
    for idx in range(len(questions)):
        for token in bigram[questions[idx]]:
            if '_' in token:
                # Token is a bigram, add to document.
                questions[idx].append(token)

    # Create a dictionary representation of the documents.
    dictionary = Dictionary(questions)

    # Filter out words that occur less than 20 documents, or more than 50% of the documents.
    dictionary.filter_extremes(no_below=20, no_above=0.20)
    corpus = [dictionary.doc2bow(q) for q in questions]


    # Set training parameters.
    num_topics = 20
    #chunksize = 2000
    passes = 15
    iterations = 5000
    #eval_every = None  # Don't evaluate model perplexity, takes too much time.

    # Make an index to word dictionary.
    temp = dictionary[0]  # This is only to "load" the dictionary.
    id2word = dictionary.id2token

    model = LdaModel(
        corpus=corpus,
        id2word=id2word,
        num_topics=num_topics,
        passes=passes,
        iterations=iterations
    )
    top_topics = model.top_topics(corpus)
    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    avg_topic_coherence = sum([t[1] for t in top_topics]) / num_topics
    print('Average topic coherence: %.4f.' % avg_topic_coherence)
    print("Top_topics")
    pprint(top_topics)
    # Step 5: Interpret Topics.
    topics = model.print_topics(num_words=50)
    print("Topics")
    pprint(topics)

    # Step 6: Assign Topics to Datasets.
    topics = [model[doc] for doc in corpus]
    dominant  = lambda x: max(x, key=lambda item: item[1])[0]
    # Step 7: Categorize Datasets.
    dominant_topic = np.array([dominant(t) for t in topics])
    dominant_topic= dominant_topic.astype(int)
    # Visualization (Optional):
    # You can visualize the results using various libraries such as matplotlib or seaborn.
    print("Dominant Topic: ", len(dominant_topic) ,dominant_topic)
    # Example: Bar chart of dominant topics
    value, topic_index, topic_counts = np.unique(dominant_topic, return_index=True,return_counts=True)
    print("Dominant Topic Index",value)
    print("topic_index: ", len(topic_index) ,"=>",topic_index)
    print("topic_counts: ", len(topic_counts) ,"=>",topic_counts)
    plt.bar(value, topic_counts)
    plt.xticks(value)
    plt.xlabel('Dominant Topic')
    plt.ylabel('Number of Datasets')
    plt.title('Distribution of Dominant Topics in Datasets')
    plt.savefig("Test.png")

main()
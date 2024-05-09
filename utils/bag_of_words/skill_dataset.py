#import nltk
#nltk.download('wordnet')
import matplotlib.pyplot as plt
import numpy as np 
import torch
import seaborn as sns
from collections import Counter
from sklearn.preprocessing import MinMaxScaler

def set_random_seed(seed=0):
    #random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

#create frequency matrix given list of skills for each dataset. Create an adj matrix for skills vs dataset
def frequencyMatrix(data):
    # Create a list of all unique skills
    all_skills = sorted(set(skill for skills_list in data.values() for skill in skills_list))

    # Create a matrix where each row represents an item and each column represents a skill
    matrix = np.zeros((len(data), len(all_skills)), dtype=int)

    # Fill the matrix with the frequencies of each skill for each item
    for i, (item, skills_list) in enumerate(data.items()):
        for skill in skills_list:
            j = all_skills.index(skill)
            matrix[i, j] += 1
    return matrix


#filter the skill for reach dataset by certain threshold
def filterData(data, threshold):
    from collections import Counter
    uniquedata ={}
    for d in data:
        uniquedata[d] = np.unique(data[d])
    # Combine all items into a single list of words
    all_words = [word for skills_list in uniquedata.values() for word in skills_list]

    # Count the frequency of each word
    word_counts = Counter(all_words)

    # Calculate the threshold for words to be removed (appearing in more than 90% of the data)
    threshold = len(uniquedata) * threshold

    # Identify common words to be removed
    common_words = [word for word, count in word_counts.items() if count > threshold or count == 1]

    # Remove common words from each item
    filtered_data = {key: [word for word in value if word not in common_words] for key, value in data.items()}
    return filtered_data

def create_plot_bog_skills(skillList, dataset_list, plot=True):
    scaler = MinMaxScaler()
    unique_items = sorted(set(item for sublist in skillList.values() for item in sublist))
    frequency =[]
    for data in dataset_list:
        item_counter = Counter(skillList[data])
        # Filter counts based on the unique items and store them
        frequency.append([item_counter.get(item, 0) for item in unique_items])
    frequency = scaler.fit_transform(frequency)
    if plot:
        plt.figure(figsize=(20,10))
        cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
        #g = sns.heatmap(bag_o_words_neuron,cmap="crest", cbar=False,xticklabels=layer_modules_label,yticklabels=[name.split('/')[-1] for name in dataset_name],vmax=1,vmin=0)
        g = sns.heatmap(frequency,cmap=cmap, cbar=False,vmax=1,vmin=0)
        #g.set_xticklabels(g.get_xticklabels(), rotation = 27)
        plt.title(f"Dataset vs Skills")
        plt.plot()
    return np.array(frequency), unique_items

def td_idf_filter(data, threshold = 0.01):
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Example list of documents (replace this with your actual datasets' skills)
    documents = [" ".join(data[dataset_name]) for dataset_name in data]
    # Initialize a TF-IDF Vectorizer
    vectorizer = TfidfVectorizer()

    # Fit and transform the documents
    tfidf_matrix = vectorizer.fit_transform(documents)

    # Get feature names to use as dataframe column headers
    feature_names = vectorizer.get_feature_names_out()

    # Calculate average TF-IDF score for each skill across all documents
    average_tfidf_scores = np.mean(tfidf_matrix.toarray(), axis=0)

    # Combine feature names with their average scores
    tfidf_scores = zip(feature_names, average_tfidf_scores)

    '''# Sort skills by their average TF-IDF score
    sorted_skills_by_tfidf = sorted(tfidf_scores, key=lambda x: x[1])

    # Display skills starting from the lowest average TF-IDF score
    for s in sorted_skills_by_tfidf:
        print(s)'''
    filtered_skill = [skills for skills, weight in tfidf_scores if weight <= 0.1]

    new_document = { dataset_name: [ skills for skills in data[dataset_name] if skills in filtered_skill]  for dataset_name in data}
    return new_document
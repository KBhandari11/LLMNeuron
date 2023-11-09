import json
import argparse
import numpy as np
import matplotlib.pyplot as plt

def pad_list_of_lists(list_of_lists, pad_value=None, max_length=None):
    if max_length is None:
        max_length = max(len(lst) for lst in list_of_lists)
    if pad_value is None:
        pad_value = min(len(lst) for lst in list_of_lists)
    padded_list = [lst + [pad_value] * (max_length - len(lst)) for lst in list_of_lists]
    return padded_list

def loadFile(path):
    f = open(path)
    data = json.load(f)
    dataList = []
    for d in data:
        dataList.append(data[d])
    return pad_list_of_lists(dataList)

def stdmtx(X):
    means = X.mean(axis =1)
    stds = X.std(axis= 1, ddof=1)
    X= X - means[:, np.newaxis]
    X= X / stds[:, np.newaxis]
    return np.nan_to_num(X)

def displayData(data,filename):
    '''fig, ax = plt.subplots()
    min_val, max_val = 0, 15'''

    x = np.array(data)
    print(np.min(x),np.max(x)) 
    intersection_matrix = (x-np.min(x))/(np.max(x)-np.min(x))
    print(np.min(intersection_matrix),np.max(intersection_matrix)) 
    '''for i in range(len(intersection_matrix)):
        print(np.min(intersection_matrix[i,:]),np.max(intersection_matrix[i,:]))'''
    '''ax.matshow(intersection_matrix, cmap=plt.cm.Blues)
    fig.tight_layout()
    plt.savefig("figure.png")
    plt.imshow(intersection_matrix, aspect='auto')
    plt.savefig("figure1.png")'''
    plt.imshow(intersection_matrix,cmap=plt.cm.RdBu, aspect='auto')
    plt.savefig(f"{filename.split('.')[0]}.png")
    '''plt.matshow(intersection_matrix,cmap=plt.cm.RdBu)
    plt.gca().set_aspect('auto')
    plt.savefig("figure3.png", dpi=600)'''


def main(args):
    data = loadFile(args.path)
    displayData(data, args.path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default="neurons_commonsense_qa.json")
    args = parser.parse_args()
    main(args)

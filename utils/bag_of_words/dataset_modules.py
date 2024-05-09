import matplotlib.pyplot as plt
import numpy as np 
import seaborn as sns
from utils.bag_of_words.sparsification import spectral_sparsification

#given the sparsity value of the model, return dataset vs modules data.
def create_plot_bog_modules(distribution, original_distribution, dataset_list,pruner_style="block", pruner_ratio="15",norm="|W|_inf", alpha=None,plot=True, random_seed=True):
    modules=["attn.q", "attn.k", "attn.v", "attn.o","gate","mlp.up", "mlp.down"]
    layer_modules_label=[ str(i)+"_"+m  for i in range(3,31) for m in modules]
    bag_o_words_neuron = []
    for dataset_name in dataset_list:
        dataset_name = dataset_name.split('/')[-1]
        #get distribution of both
        pruned_data = distribution[pruner_style][pruner_ratio][dataset_name][norm][3:31,:].flatten()
        original_data = np.array(original_distribution["distribution"][norm])[3:31,:].flatten()

        #take the difference and normalize
        pruned_acc = distribution[pruner_style][pruner_ratio][dataset_name]["Accuracy"][2]
        original_acc = original_distribution[dataset_name][2]

        sparsity_ratio = [(((w_pruned)/w_org)) for w_org, w_pruned  in zip(original_data,pruned_data) ]
        min_weight_reduction = min(sparsity_ratio)
        range_weight_reduction = max(sparsity_ratio) - min_weight_reduction
        data = [(1-abs(original_acc-pruned_acc))*(((weight - min_weight_reduction) / range_weight_reduction)) for weight in sparsity_ratio ]
        
        bag_o_words_neuron.append(data)
    bag_o_words_neuron =  np.array(bag_o_words_neuron)
    if alpha != None:
        '''flattened_array= np.sort(bag_o_words_neuron.flatten())
        quartile_value= np.percentile(flattened_array,alpha)
        bag_o_words_neuron[bag_o_words_neuron < quartile_value] = 0'''
        original_array = bag_o_words_neuron
        bag_o_words_neuron = spectral_sparsification(bag_o_words_neuron, alpha, random_seed=random_seed)

    else:
        original_array = bag_o_words_neuron
    if plot:
        fig, axis = plt.subplots(figsize=(40,10),ncols=2)
        '''count, bins, ignored = axis[1].hist(flattened_array, 100, density=False, alpha=0.9,color='blue', edgecolor='black')
        fig.suptitle(f'Pruner Strategy {pruner_style} and ratio {pruner_ratio} with cutoff at {int(alpha)}th Percentile')
        axis[1].fill_betweenx(y=[0, max(count)], x1=min(bins), x2=quartile_value, color='red', alpha=0.3)
        axis[1].axvline(quartile_value, color='r', linestyle='--', label=f'{int(alpha)}th percentile = {quartile_value}')
        axis[1].set_xlabel('imporance of module')
        axis[1].set_ylabel('frequency')
        axis[1].set_title(f'Distribution of Importance of Neurons')
        axis[1].legend()'''
        cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
        #g = sns.heatmap(bag_o_words_neuron,cmap="crest", cbar=False,xticklabels=layer_modules_label,yticklabels=[name.split('/')[-1] for name in dataset_name],vmax=1,vmin=0)
        g = sns.heatmap(original_array,cmap=cmap,ax=axis[0], cbar=False)
        g.set_xticklabels(g.get_xticklabels(), rotation = 27)
        axis[0].set_title(f"Original Adjacency Matrix")
        cmap = sns.color_palette("ch:start=.2,rot=-.3", as_cmap=True)
        #g = sns.heatmap(bag_o_words_neuron,cmap="crest", cbar=False,xticklabels=layer_modules_label,yticklabels=[name.split('/')[-1] for name in dataset_name],vmax=1,vmin=0)
        g = sns.heatmap(bag_o_words_neuron,cmap=cmap,ax=axis[1], cbar=False)
        g.set_xticklabels(g.get_xticklabels(), rotation = 27)
        axis[1].set_title(f"Pruned Adjacency Matrix")
        plt.show()
        
    return np.array(bag_o_words_neuron), layer_modules_label
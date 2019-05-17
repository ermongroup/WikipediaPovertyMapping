#!/usr/bin/env python3
'''
Author: Evan Sheehan
Project: Stanford AI Lab, Stefano Ermon Group & Computational Sustainability, Wiki-Satellite


'''

import sys
sys.path.append("/atlas/u/esheehan/wikipedia_project/dataset/text_dataset/dataset_modules")
from data_processor import *
import numpy as np
import gensim
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

PATH = "/atlas/u/esheehan/wikipedia_project/RNN/Doc2Vec/models/wikimodel_DBOW_vector300_window8_count15_epoch10/wikimodel.doc2vec"

if __name__ == '__main__':
    
    array = load_coordinate_array("full", uncategorized=True, verbose=True)
    model = gensim.models.Doc2Vec.load(PATH)
    print("Loaded Doc2Vec model...")

    tsne(array, model)

def tsne(array, model, perp=50, rate=200, n=2500, r=0, test=100, verbose=False):
   
    embeddings = []
    labels = []
    for i, article in enumerate(array):
        embeddings.append(model.docvecs[get_title(article)])
        labels.append(get_title(article))
    print("Created embeddings array...")

    tsne_model = TSNE(perplexity=perp, learning_rate=rate, n_components=2, init='pca', n_iter=n, random_state=r, verbose=verbose)
    new_values = tsne_model.fit_transform(embeddings)
    
    print("Fit model...")

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])

    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate((labels[i] if i % test == 0 else ""),
            xy=(x[i], y[i]),
            xytext=(5, 2),
            textcoords='offset points', ha='right', va='bottom')
    plt.show()

    plt.savefig(CONST_PREFIX + "maps/tsne.png")
    print("Plot " + CONST_PREFIX + "maps/tsne.png " + "saved!")



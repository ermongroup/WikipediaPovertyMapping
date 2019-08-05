#!/usr/bin/env python3
'''
Author: Evan Sheehan
Project: Stanford AI Lab, Stefano Ermon Group & Computational Sustainability, Wiki-Satellite

Loads the array of coordinate articles and trains a doc2vec model on them
'''

import sys
from data_processor import *
import gensim
import os
import collections
import random
import multiprocessing

def read_corpus(array):
    data = []
    for i in array:
        #if "Body Text" in array[i][6]:
            #doc = array[i][6]["Body Text"]
        #else:
            #doc = array[i][1]
        doc = get_text(i)
        data.append(gensim.models.doc2vec.TaggedDocument(gensim.utils.simple_preprocess(doc), [get_title(i)]))
    return data

if __name__ == '__main__':

    # Load all corodinate articles across all categories, including Uncategorized ones
    wiki_train_file = load_coordinate_array("full", uncategorized=True,  verbose=True)
    train_corpus = read_corpus(wiki_train_file)
    print("Data formatted")

    # Instantiate the model and build the vocabulary
    # DBOW, with word vectors as well
    model = gensim.models.doc2vec.Doc2Vec(dm=0, dbow_words=1, vector_size=300, window=8, min_count=15, epochs=10, workers=multiprocessing.cpu_count())
    # DM
    #model = gensim.models.doc2vec.Doc2Vec(vector_size=1000, window=8, min_count=15, epochs=50, workers=multiprocessing.cpu_count())
    model.build_vocab(train_corpus)
    print("Vocabulary built")

    # Train the model
    model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)
    print("Model trained")

    # Save the model
    model.save("../models/wikimodel_DBOW_vector300_window8_count15_epoch10/wikimodel.doc2vec")
    print("Model saved")


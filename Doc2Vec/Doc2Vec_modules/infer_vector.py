import  numpy as np
import gensim

path = "/atlas/u/esheehan/wikipedia_project/RNN/Doc2Vec/models/wikimodel_DBOW_vector300_window8_count15_epoch10"
model = gensim.models.Doc2Vec.load(path)

data = # load docs

vecs = []
for i in data:
    vecs.append(model.infer_vector(gensim.utils.simple_preprocess(i)))

np.save("/atlas/u/esheehan/inferred_vectors", vecs)

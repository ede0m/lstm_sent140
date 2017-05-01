import dataset as ds
import gensim 
import numpy
# import pydot
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.utils import plot_model

# Load W2V Model #
w2v = gensim.models.Word2Vec.load('../vocab')
numWords = len(w2v.wv.vocab)
dimension = w2v.vector_size


# Set up embeddings #
# embedding_weights = numpy.zeros((numWords, dimension))
# count = 0
# for word in w2v.wv.vocab.items():
# 	word = word[0]
# 	embedding_weights[count,:] = w2v[word]
# 	count += 1


# Set up Training Data # 
d = ds.DataSet('../../lstm_data/training.csv', w2v, 2, dimension) ## 2 = GRAM VALUE HERE 
data = d.data_padded
targets = d.outputs
input_len = len(data[0])
feats = len(data[0][0])
# print(input_len, " ", feats)


# Set up Classifier Model #
model = Sequential()
# embedding (one-hot) layer #
#
#     - current assumption is  input_len=2 for two word embeddings. One in each LSTM cell. 
# model.add(Embedding(input_dim=numWords, output_dim=dimension, mask_zero=True, weights=[embedding_weights])) # might need to be numWords + 1

model.add(LSTM(100, input_shape=(input_len, feats), dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# plot_model(model, to_file='../model.png')
import sys
import numpy as np
import pandas as pd
import string
import collections
from keras.models import Sequential
from keras.layers import LSTM, Bidirectional, Dropout, Dense, Activation
from keras.optimizers import RMSprop, Adadelta, Adam
from keras.models import load_model
from keras.callbacks import ReduceLROnPlateau
from keras import callbacks
from keras.utils import multi_gpu_model

# model = load_model('/any/previous/model.h5')

puncts = string.punctuation.replace('.','')
punct = str.maketrans('','', puncts)

data = open('story.txt','r').read()

# data pre-processing only for alphabet strings
def clean(xx):
    return ' '.join(x for x in xx.split() if not x.isnumeric())
    
cdata = clean(data.translate(punct))

sents = [s for s in cdata.split()]

vocab = sorted(collections.Counter(sents))
vocab2idx = {v:idx for idx,v in enumerate(vocab)}
idx2vocab = {idx:v for idx,v in enumerate(vocab)}

seq_len = 10
seq_step = 1
sequences = []
nextword = []

for idx in range(len(sents) - seq_len):
    seq_sent = sents[idx : idx + seq_len]
    nxt_word = sents[idx + seq_len]
    sequences.append(seq_sent)
    nextword.append(nxt_word)

seq = pd.DataFrame({'sequence':sequences, 'target':nextword})

sequence_arr = np.zeros((len(seq), seq_len, len(vocab)), dtype=bool)
target_arr = np.zeros((len(seq), len(vocab)), dtype=bool)

for s_idx,x,y in seq.itertuples(index=True):
    target_arr[s_idx][vocab2idx[y]] = 1
    for w_idx,word in enumerate(x):
        sequence_arr[s_idx][w_idx][vocab2idx[word]] = 1       

model = Sequential()
model.add(Bidirectional(LSTM(256, activation='relu'), input_shape=(seq_len, len(vocab))))
model.add(Dropout(0.4))
model.add(Dense(len(vocab)))
model.add(Activation('softmax'))

parallel_model = multi_gpu_model(model, gpus=4)
parallel_model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['categorical_accuracy'])

lr_reducer = ReduceLROnPlateau(monitor='val_loss', patience=1, verbose=1, factor=0.5, min_lr=0.01)
batch_size = 1024
num_epochs = 20

print(parallel_model.summary())

def onehot2word(arr):
    return idx2vocab[arr.argmax()]

def word2onehot(word):
    vidx = vocab2idx[word]
    varr = np.zeros((1, len(vocab)), dtype=bool)
    varr[0, vidx] = 1
    return varr

def prob2onehot(prob):
    foo = np.zeros((1, len(vocab)), dtype=bool)
    foo[0, prob.argmax()] = 1
    return foo

def headstart():
    hswords = []
    for w in 'Hobbits lived in the woods happily and the story begins'.split():
        hswords.append(word2onehot(w))
    return np.array(hswords).transpose(1,0,2)

def gen_text(model, word_limit):
    textcum = []
    text_generated = []
    sequence_arr = headstart()
    text_generated.extend([x for y in sequence_arr for x in y])
    for idx in range(word_limit):
        predicted_arr = prob2onehot(model.predict(sequence_arr))
        text_generated.append(predicted_arr)
        sequence_arr = np.concatenate((sequence_arr[0, 1:, :], predicted_arr)).reshape(sequence_arr.shape)
    for w in text_generated:
        textcum.append(onehot2word(w))
    return ' '.join(textcum)

print("Uninitialised generated text :", gen_text(parallel_model, 200))

for _ in range(num_epochs):
    parallel_model.fit(sequence_arr, target_arr, batch_size=batch_size, epochs=1, callbacks=[lr_reducer], validation_split=0.10)
    print("Generated text :", gen_text(parallel_model, 200))

model.save('/path/mymodel.h5')
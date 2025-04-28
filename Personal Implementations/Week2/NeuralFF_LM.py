#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Created on Mon April 28 9:22:27 2025
    @author: STRH
    Continuing my journey to learn natural language processing.
    This is a simple implementation of Neural Language Model (Simple Feed Forward Neural Network)
    for language modeling.
"""

#%% Import Libraries
import numpy as np
import string
import pickle
import matplotlib.pyplot as plt
from IPython.display import clear_output

import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

#%% Function to read Hamshahri corpus
def read_hamshahri_corpus(path):
    count = 0
    res = []
    corpus = ''

    with open(path, 'r') as ptr:
        for line in ptr:
            res.append(line)

    for i in range(0, len(res)):
        if count>500:
            break
        if res[i].startswith(".DID"):
            count += 1
            continue

        if res[i].startswith(".Date"):
            continue

        if res[i].startswith(".Cat"):
            continue

        corpus = corpus+ ' ' + res[i].strip()

    return corpus

#%% Function that returns an array of words that occured less than N time in given corpus
def get_rare_words(corpus, N):
    tokens = corpus.split()

    dictionary = {}
    for word in tokens:
        if word in dictionary:
            dictionary[word] += 1
        else:
            dictionary[word] = 1

    rare_words = []
    for key, value in dictionary.items():
        if value <= N:
            rare_words.append(key)

    print('count of rare words: ', len(rare_words))
    return rare_words

#%% Remove punctuation and non-alphabetic tokens from corpus and lowercase the tokens
def clean_corpus(sentence):
    tokens = sentence.split()
    translation_table = str.maketrans('', '', "\"#$%&'()*+,-/:;<=>?@[\]^_`{|}~،؟!❊#$٪^&*)(ـ+=-؛:{}")
    tokens = [w.translate(translation_table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in rare_words]
    return tokens

#%% apply function above to the corpus
USE_SAVED_OBJ = True

# path to dataset
data_path = "/home/ai/Vision/Khosravi/LLMs/Datasets/HamshahriOldCorpus/Corpus/Hamshahri-Corpus.txt"

if (not USE_SAVED_OBJ):
    # read the corpus as a string
    corpus_raw = read_hamshahri_corpus(data_path)

    # extract the rare words
    rare_words =  get_rare_words(corpus_raw, 1)

    # split the input corpus into sentences by '.'
    sentences = corpus_raw.split('.')

    tokenized_sentences = []

    # clean each sentence and split it into tokens
    for sentence in sentences:
        tokens = clean_corpus(sentence)
        tokenized_sentences.append(tokens)

    # save the cleaned corpus in pickle file
    file = open("./tokenized_sentences.obj", "wb")
    pickle.dump(tokenized_sentences, file)
    file.close()

else:
    # load the cleaned sentences by pickle from file
    file = open("./tokenized_sentences.obj", "rb")
    tokenized_sentences = pickle.load(file)
    file.close()

# print(corpus_raw)
print(len(tokenized_sentences))

#%% Extract sequences of N-grams from the corpus
# count of given words to predict the next word 
window_size = 3

#length of sequence or n-gram
seq_length = window_size + 1

# store the n-grams in sequences list
sequences = []

# extract a sentence of each seq_length consecutve words from given sentences.
for tokens in tokenized_sentences:
    for i in range(seq_length, len(tokens)):
        seq = tokens[i-seq_length:i]
        line = ' '.join(seq)
        sequences.append(line)

print("Total sequences: %d" % len(sequences))
print(sequences[15])

#%% Tokenize the sequences
# create a tokenizer object and fit it on the sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sequences)

# tokenize the sequences into encoded numbers
sequences = tokenizer.texts_to_sequences(sequences)
vocab_size = len(tokenizer.word_index) + 1
sequences = np.array(sequences)

# split the sequences into input (x) and output (y)
x,y = sequences[:,:-1], sequences[:,-1]

print(sequences[15])
print(vocab_size)

#%% Convert the output into categorical (one-hot) format requiered for training the model
y = to_categorical(y, num_classes=vocab_size)
print(np.shape(y[15]))
print(y[15])
print(np.shape(x))

#%% Split data into train, test and validation sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42)
length = int(len(x_test)*0.5)

y_validation = y_test[:length]
x_validation = x_test[:length]

y_test = y_test[length:2*length]
x_test = x_test[length:2*length]

#%% Define plot losses callback
class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []

        self.fig = plt.figure()

        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1

        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.show();

plot_losses = PlotLosses()

#%% Define the model architecture
model = Sequential()
model.add(Embedding(vocab_size, 50, input_length=window_size, name='Embedding-layer'))
model.add(Flatten())
model.add(Dense(int(vocab_size/2), activation='relu', name='hidden-layer'))
model.add(Dense(vocab_size, activation='softmax', name='output-layer'))

# print summary of the model architecture
print(model.summary())


#%% Train model
checkpoint = ModelCheckpoint('./model-{epoch:03d}---{val_accuracy:.4f}.weights.h5', monitor='val_loss', verbose=1, 
                             save_best_only=True, save_weights_only=True, mode='min')

# compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# train model
model.fit(x_train, y_train, 
          batch_size=50, 
          epochs=10, 
          validation_data=(x_validation, y_validation), 
          callbacks=[checkpoint, plot_losses])

#%% Load and test the model
best_model_weights = "model-002---0.1129.weights.h5"
model.load_weights(best_model_weights)

# predict the class of test data
res = model.predict_classses(x_test[0:200])
# %% given the ID od word returns its corresponding word
def convert_ID_to_word(ID):
    for word, index in tokenizer.word_index.items():
        if index == ID:
            return word
    return

# given the input sentense as an array of word ids, returns the string of sentence
def get_sentence_from_IDs(x):
    sentence = ''
    for element in x:
        sentence += convert_ID_to_word(element) + ' '
    return sentence

# run above functions
arr = [0, 6, 13, 17, 18, 22, 35, 53, 73, 105, 112, 124, 128, 137, 145, 154, 161, 185]
for i in arr:
    print(get_sentence_from_IDs(x_test[i]), " : ", convert_ID_to_word(res[i]))
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

from keras.preprocessing.text import Tokenizer

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
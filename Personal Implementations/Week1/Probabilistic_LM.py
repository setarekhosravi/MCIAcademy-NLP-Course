#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
    Created on April 23 17:28 2025
    @author: STRH
    Starting my journey into the world of Natural Language Processing (NLP) and Large Language Models (LLMs).
    This script is a simple implementation of a probabilistic language model using Python.
    Unigram and bigram models are implemented to demonstrate the basic concepts of language modeling.
    Analysis of Persian poetry dataset.
"""

#%% importing the necessary libraries
import math

#%% reading the data
def read_file(file_path):
    data = []
    with open(file_path, 'r') as ptr:
        for line in ptr:
            tmp = line.strip()
            tmp = tmp.replace('؟', '')
            tmp = tmp.replace('.', '')
            tmp = tmp.replace('!', '')
            tmp = tmp.replace('،', '')
            data.append(tmp)

    return data

sentences = read_file('train.txt')
# print(sentences[0])
# print(sentences)
# print(len(sentences))

#%% Unigram Model
def calculate_unigram_probabilities(corpus):
    word_count = {}
    total_words = 0

    for doc in corpus:
        tmp = doc.split(" ")
        total_words += len(tmp)

        for word in tmp:
            if (not (word in word_count)):
                word_count[word] = 0
            word_count[word] += 1

    return word_count, total_words

word_count, total_words = calculate_unigram_probabilities(sentences)
# print(word_count["ستاره"])
# print(total_words)

#%% Unigram probabilities
def get_unigram_probabilities(word):
    # prob = math.log(word_count[word] / total_words)
    prob = (word_count[word] / total_words)
    return prob

#%% Unigram Language Model
def unigram_LM(sentence):
    prob = 1
    # prob = 0
    tmp = sentence.split(" ")
    for word in tmp:
        prob *= get_unigram_probabilities(word)
        # prob += get_unigram_probabilities(word)

    return prob

prob_1 = unigram_LM("عکس هر نقشی نتابد تا ابد")
prob_2 = unigram_LM("عکس هر محکم نتابد تا ابد")

if prob_1 > prob_2:
    print("Sentence 1 is more probable")
else:
    print("Sentence 2 is more probable")

#%% Bigram Model
def collect_bigrams(corpus):
    word_bigrams = {}

    for doc in corpus:
        tmp = doc.split(" ")
        for i in range(1, len(tmp)):
            word = tmp[i]
            prev_word = tmp[i - 1]

            if not (word in word_bigrams):
                word_bigrams[word] = []

            word_bigrams[word].append(prev_word)

    return word_bigrams

word_bigrams = collect_bigrams(sentences)
print(word_bigrams["ستاره"])
print(len(word_bigrams["ستاره"]))

#%% Bigram probabilities
def get_bigram_probabilities(word, prev_word):
    bigrams = word_bigrams[word]
    count = 0

    for element in bigrams:
        if element == prev_word:
            count += 1

    N = word_count[prev_word]
    prob = count / N

    return prob

print(get_bigram_probabilities("ستاره", "آسمان"))

#%% Bigram Language Model
def bigram_LM(sentence):
    prob = 1
    tmp = sentence.split(" ")
    
    prob *= word_count[tmp[0]] / total_words # unigram probability of the first word

    for i in range(1, len(tmp)):
        prob *= get_bigram_probabilities(tmp[i], tmp[i - 1]) # bigram probability for other words

    return prob

prob_1 = bigram_LM("عکس هر نقشی نتابد تا ابد")
prob_2 = bigram_LM("عکس هر محکم نتابد تا ابد")

if prob_1 > prob_2:
    print("Sentence 1 is more probable")
else:
    print("Sentence 2 is more probable")

#%% Conclusions
"""
    Bigram model is more accurate than Unigram model.
"""
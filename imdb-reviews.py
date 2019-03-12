######################
# Load Libraries

import numpy as np
import os
import glob
import sys
import json
import pickle as pkl
from collections import Counter
import matplotlib.pyplot as plt

######################

######################
# Define preprocessing functions

def create_dict(review):
    d = dict()
    for word in review:
        if word in d.keys():
            d[word] += 1.0
        else:
            d[word] = 1.0
    return d

def remove_digits(word):
    s = set()
    digits = set(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
    for char in word:
        s.add(char)
    return len(s & digits) != 0   

def remove_chars(string):
    string = string.replace('!', '').replace(',', '').replace('.', '').replace('?', '').replace(';', '').replace('"', '')
    string = string.replace(':', '').replace('+', '').replace('=', '').replace('$', '').replace('&', '').replace('-', '')
    string = string.replace('(', '').replace(')', '').replace('#', '').replace('*', '').replace('%', '').replace('/', '')
    string = string.replace('<', '').replace('>', '').replace('@', '').replace('[', '').replace(']', '')
    return string
    
def preprocess_review(string):
    string = string.lower()
    string = remove_chars(string)
    words = string.split()
    
    for i in range(len(words)-1, -1, -1):
        if words[i] in stop_words or remove_digits(words[i]) or len(words[i]) < 4 or not words[i].isalnum():
            del words[i]
    
    return create_dict(words)

def preprocess_review_test(string):
    string = string.lower()
    string = remove_chars(string)
    words = string.split()
    
    for i in range(len(words)-1, -1, -1):
        if words[i] in stop_words or remove_digits(words[i]) or words[i] not in vocab or len(words[i]) <4  or not words[i].isalnum():
            del words[i]
    
    return create_dict(words)

def cosine_distance(word1, word2):
    rep1 = [0]*26
    rep2 = [0]*26
    
    for i in range(len(word1)):
        rep1[ord(word1[i]) - 97] += 1.0
    for j in range(len(word2)):
        rep2[ord(word2[j]) - 97] += 1.0
        
    w = sum(i * j for i, j in zip(rep1, rep2))
    a = sum(i * i for i in rep1)
    b = sum(j * j for j in rep2)
    a = np.sqrt(a)
    b = np.sqrt(b)
    
    return (w / (a * b)) > 0.85

stop_words = set(['a', 'am', 'all', 'an', 'and', 'any', 'also', 'are', 'aren\'t', 'after',
              'as','at', 'above', 'about', 'be', 'between','been', 'but', 'below', 'because','by', 'can','can\'t',
              'cannot', 'both', 'could', 'day', 'do','does', 'don\'t', 'did', 'doing', 'didn\'t',
              'for','four', 'from', 'go', 'got', 'going', 'he', 'hotel', 'had', 'have',
              'his', 'her', 'here', 'has','having', 'him', 'how', 'i', 'i\'d', 'i\'ll', 'i\'m', 'i\'ve', 'if', 'it\'s', 'is',
              'into', 'its','it', 'in', 'isn\'t','itself', 'just', 'long', 'lets',
              'let\'s', 'me', 'more', 'most', 'my', 'myself', 'movie', 'no', 'not',
              'nor', 'on', 'off', 'or', 'of', 'one', 'oh', 'once', 'only', 'our', 'out', 'over',
              'own', 'same', 'she','should', 'she\'d', 'so', 'some', 'such','than', 'that', 'the',
              'that\'s', 'room', 'there', 'their', 'theirs', 'they', 'they\'d', 'they\'ve', 'they\'re','this', 'them', 'then',
              'two', 'three', 'those', 'these', 'thing', 'things', 'through', 'to', 'too', 'under', 'until', 'up', 'us',
              'upon', 'very', 'was', 'went', 'wasn\'t', 'we', 'we\'d', 'we\'ll', 'we\'ve', 'we\'re', 'will', 'were',
              'weren\'t', 'what', 'what\'s', 'when', 'where', 'where\'s', 'which', 'while','who', 'whom','who\'s',
              'why', 'why\'s', 'with', 'without', 'won\'t', 'would', 'wouldn\'t', 'you', 'you\'d', 'you\'ll',
              'you\'re', 'you\'ve', 'your', 'yours'])

######################

######################
# Read and store training data

input_path = '../imdb_reviews/train/'

positive_files = os.listdir(input_path + 'pos/')
negative_files = os.listdir(input_path + 'neg/')

positive_reviews = []
negative_reviews = []

for i in range(len(positive_files)):
    f = open(input_path + 'pos/' + positive_files[i])
    a = f.readline()
    w = preprocess_review(a)
    f.close()
    positive_reviews.append(w)
    
for i in range(len(negative_files)):
    f = open(input_path + 'neg/' + negative_files[i])
    a = f.readline()
    w = preprocess_review(a)
    f.close()
    negative_reviews.append(w)

######################

######################
# Preprocess data

vocab = set()
vocab_count = Counter(dict())
all_reviews = negative_reviews + positive_reviews
for review in all_reviews:
    vocab_count += Counter(review)
vocab = set(vocab_count.keys())

frequencies = np.array(vocab_count.values())
mean = np.mean(frequencies)
std = np.std(frequencies)
print mean, std

fig = plt.hist(vocab_count.values())
plt.show()

fig = plt.boxplot(vocab_count.values())
plt.show()

words_to_remove = set()
for key in vocab_count.keys():
    if vocab_count[key] < 3 or vocab_count[key] > mean + 10*std:
        words_to_remove.add(key)
        del vocab_count[key]
        vocab.remove(key)

frequencies = np.array(vocab_count.values())
mean = np.mean(frequencies)
std = np.std(frequencies)

frequencies = np.array(vocab_count.values())
mean = np.mean(frequencies)
std = np.std(frequencies)
print mean, std

for review in negative_reviews:
    for each in list(words_to_remove):
        if each in review.keys():
            del review[each]
            
for review in positive_reviews:
    for each in list(words_to_remove):
        if each in review.keys():
            del review[each]

######################

######################
# Build model for training by estimating parameters for Naive Bayes

all_reviews = []
all_reviews = negative_reviews + positive_reviews

p_positive = -np.log2(float(len(positive_reviews))/(len(all_reviews)))
p_negative = -np.log2(float(len(negative_reviews))/(len(all_reviews)))
print p_negative, p_positive

count_positive = Counter(dict())
count_negative = Counter(dict())

for review in negative_reviews:
    count_negative += Counter(review)
    
for review in positive_reviews:
    count_positive += Counter(review)

vocab_positive = set(count_positive.keys())
vocab_negative = set(count_negative.keys())

sum_positive = sum(count_positive.values())
sum_negative = sum(count_negative.values())

len_positive = len(vocab_positive)
len_negative = len(vocab_negative)

len_vocab = len(vocab)

parameters_positive = dict()
parameters_negative = dict()

for word in vocab:
    if word in vocab_negative:
        p = (count_negative[word] + 1)/(sum_negative + len_vocab)
        parameters_negative[word] = (-np.log2(p), -np.log2(1-p))

    else:
        p = 1/(sum_negative + len_vocab)
        parameters_negative[word] = (-np.log2(p), -np.log2(1-p))


    if word in vocab_positive:
        p = (count_positive[word] + 1)/(sum_positive + len_vocab)
        parameters_positive[word] = (-np.log2(p), -np.log2(1-p))

    else:
        p = 1/(sum_positive + len_vocab)
        parameters_positive[word] = (-np.log2(p), -np.log2(1-p))

all_parameters = [parameters_negative, parameters_positive, p_negative, p_positive, list(vocab)]

######################

######################
# Save model parameters in pkl file

f = open('imdb_model.pkl', 'w')
pkl.dump(all_parameters, f)
f.close()

######################

######################
# Read and store validation data

test_input_path = '../imdb_reviews/test/'

positive_files_test = os.listdir(test_input_path + 'pos/')
negative_files_test = os.listdir(test_input_path + 'neg/')

positive_reviews_test = []
negative_reviews_test = []

for i in range(len(positive_files_test)):
    f = open(test_input_path + 'pos/' + positive_files_test[i])
    a = f.readline()
    w = preprocess_review_test(a)
    f.close()
    positive_reviews_test.append(w)
    
for i in range(len(negative_files_test)):
    f = open(test_input_path + 'neg/' + negative_files_test[i])
    a = f.readline()
    w = preprocess_review_test(a)
    f.close()
    negative_reviews_test.append(w)

tp = 0.0
fp = 0.0
fn = 0.0
tn = 0.0

for i in range(len(positive_reviews_test)):
    positive_score = 0.0
    negative_score = 0.0

    for word in list(vocab):
        if word in positive_reviews_test[i].keys():
            positive_score += parameters_positive[word][0]
            negative_score += parameters_negative[word][0]
            
        else:
            positive_score += parameters_positive[word][1]
            negative_score += parameters_negative[word][1]

    positive_score += p_positive
    negative_score += p_negative
    
    if positive_score <= negative_score:
        tp += 1
    else:
        fn += 1 

for i in range(len(negative_reviews_test)):
    positive_score = 0.0
    negative_score = 0.0

    for word in list(vocab):
        if word in negative_reviews_test[i].keys():
            positive_score += parameters_positive[word][0]
            negative_score += parameters_negative[word][0]
            
        else:
            positive_score += parameters_positive[word][1]
            negative_score += parameters_negative[word][1]

    positive_score += p_positive
    negative_score += p_negative
    
    if positive_score > negative_score:
        tn += 1
    else:
        fp += 1

print tp, fp, tn, fn

prec = tp / (tp + fp)
rec = tp / (tp + fn)

f1_score = 2 * prec * rec / (prec + rec)

print prec, rec, f1_score

######################





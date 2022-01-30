# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 15:57:42 2022

@author: 836666
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
#Text cleaning
import re, string
import nltk
from nltk.stem import WordNetLemmatizer,PorterStemmer
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

cb_ds = pd.read_csv('cyberbullying_tweets.csv')

cb_ds.drop_duplicates("tweet_text",inplace=True)

#Remove punctuations, links, stopwords, mentions and \r\n new line characters
def strip_all_entities(tweet_text): 
    tweet_text = tweet_text.replace('\r', '').replace('\n', ' ').lower() #remove \n and \r and lowercase
    tweet_text = re.sub(r"(?:\@|https?\://)\S+", "", tweet_text) #remove links and mentions
    tweet_text = re.sub(r'[^\x00-\x7f]',r'', tweet_text) #remove non utf8/ascii characters such as '\x9a\x91\x97\x9a\x97'
    banned_list= string.punctuation
    table = str.maketrans('', '', banned_list)
    tweet_text = tweet_text.translate(table)
    tweet_text = [word for word in tweet_text.split() if word not in stop_words]
    tweet_text = ' '.join(tweet_text)
    tweet_text =' '.join(word for word in tweet_text.split() if len(word) < 14) # remove words longer than 14 characters
    return tweet_text

def decontract(tweet_text):
    tweet_text = re.sub(r"can\'t", "can not", tweet_text)
    tweet_text = re.sub(r"n\'t", " not", tweet_text)
    tweet_text = re.sub(r"\'re", " are", tweet_text)
    tweet_text = re.sub(r"\'s", " is", tweet_text)
    tweet_text = re.sub(r"\'d", " would", tweet_text)
    tweet_text = re.sub(r"\'ll", " will", tweet_text)
    tweet_text = re.sub(r"\'t", " not", tweet_text)
    tweet_text = re.sub(r"\'ve", " have", tweet_text)
    tweet_text = re.sub(r"\'m", " am", tweet_text)
    return tweet_text

def clean_hashtags(tweet):
    new_tweet = " ".join(word.strip() for word in re.split('#(?!(?:hashtag)\b)[\w-]+(?=(?:\s+#[\w-]+)*\s*$)', tweet)) #remove last hashtags
    new_tweet2 = " ".join(word.strip() for word in re.split('#|_', new_tweet)) #remove hashtags symbol from words in the middle of the sentence
    return new_tweet2

def filter_chars(a):
    sent = []
    for word in a.split(' '):
        if ('$' in word) | ('&' in word):
            sent.append('')
        else:
            sent.append(word)
    return ' '.join(sent)

#Remove multiple sequential spaces
def remove_mult_spaces(tweet_text):
    return re.sub("\s\s+" , " ", tweet_text)

def stemmer(tweet_text):
    tokenized = nltk.word_tokenize(tweet_text)
    ps = PorterStemmer()
    return ' '.join([ps.stem(words) for words in tokenized])

def deep_clean(tweet_text):
    tweet_text = decontract(tweet_text)
    tweet_text = strip_all_entities(tweet_text)
    tweet_text = clean_hashtags(tweet_text)
    tweet_text = filter_chars(tweet_text)
    tweet_text = remove_mult_spaces(tweet_text)
    tweet_text = stemmer(tweet_text)
    return tweet_text

texts_new = []
for t in cb_ds.tweet_text:
    texts_new.append(deep_clean(t))
    
texts_new = np.asarray([texts_new])
texts_new = texts_new.reshape(46017,1)
new_data = np.hstack((texts_new, cb_ds.to_numpy()[:, 1].reshape(46017, 1)))

pd.DataFrame(new_data).to_csv("new_cyberbullying_data.csv")


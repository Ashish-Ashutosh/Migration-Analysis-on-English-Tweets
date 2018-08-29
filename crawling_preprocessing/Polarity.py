import logging
import re
import itertools
import json
from autocorrect import spell
from os.path import abspath, exists
import csv
import sys
import numpy as np
import pandas as pd
from wordsegment import load, segment
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import stopwords

logging.basicConfig(level=logging.INFO)

# csvFile = open('pre_processed_tweets.csv', 'a')
#
# fp = "./pre_processed_tweets.csv"
# with open(fp, 'rt', encoding='utf-8') as data_in:
#     for line in data_in:
#         raw_data = {'tweets':[line]}
#
#     df = pd.DataFrame(raw_data, columns=['tweets'])
#
# print(df)


# output file
# csvFile = open('SAMPLE.csv', 'a')

# dictionary
data = {'Tweet Text': [], 'index Label': []}

with open('TestPolarityInput.csv', 'r') as data_in:
    for line in data_in:
        tweet = line.split("\t")[0]

        # creating a dictionary
        data['Tweet Text'].append(tweet)
        data['index Label'].append("0")

df = pd.DataFrame(data, columns=['Tweet Text', 'Label'])
#print(df)  

# print(df)
# df_reorder = df[['index Label', 'Tweet Text']]
# print(df)
df.to_csv('PolarityTestColumn.csv', index=False)


#
# df = pd.read_csv('pre_processed_tweets.csv')
# #print(df)
# data = {'index Label' :[], 'Tweet Text': []};
# for line in df:
#     print(df[line])
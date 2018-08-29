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
from html.parser import HTMLParser
logging.basicConfig(level=logging.INFO)


def preprocess(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''

    corpus = []
    load()
    stopWords = set(stopwords.words('english'))
    data = {'Tweet Text': []};

    with open(fp, 'rt', encoding='utf-8') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
                tweet = line.split("\t")[0]

# remove emoticons
                try:
                    emoji_pattern = re.compile("["
                        u"\U0001F600-\U0001F64F" # emoticons
                        u"\U0001F300-\U0001F5FF" # symbols & pictographs
                        u"\U0001F680-\U0001F6FF" # transport & map symbols
                        u"\U0001F1E0-\U0001F1FF" # flags (iOS)
                        "]+", flags=re.UNICODE)
                    tweet = emoji_pattern.sub(r'', tweet)
                except Exception:
                    pass


                # remove unicode
                try:
                    tweet = tweet.encode('ascii').decode('unicode-escape').encode('ascii','ignore').decode("utf-8")
                except Exception:
                    pass

                # remove more unicode characters
                try:
                    tweet = tweet.encode('ascii').decode('unicode-escape')
                    tweet = HTMLParser().unescape(tweet)
                except Exception:
                    pass
                #creating a dictionary
                data['Tweet Text'].append(tweet)
    #creating a dataframe
    df = pd.DataFrame(data)
    #to order the columns in the csv file (while copying from dataframes to a CSV file)
    df_reorder = df[['Tweet Text']]
    #writing dataframe to a csv file
    #df_reorder.to_csv('semeval_preprocessed_training_data.csv', encoding='utf-8', index= False)
    df_reorder.to_csv('favor_tweets_world_semeval_preprocessed_test_data.csv', encoding='utf-8', index= False)

    return corpus

# # Contractions
# def loadAppostophesDict(fp):
#     apoDict = json.load(open(fp))
#     return apoDict

if __name__ == "__main__":
    DATASET_FP = "/home/stark/PycharmProjects/untitled/favourTweetsWorld_WithoutDuplicateRecord.csv"
    #DATASET_FP = "./SemEval2018-T3-train-taskA.txt"
    #APPOSTOPHES_FP = "./appos.txt"
    #apoDict = loadAppostophesDict(APPOSTOPHES_FP)
corpus = preprocess(DATASET_FP)


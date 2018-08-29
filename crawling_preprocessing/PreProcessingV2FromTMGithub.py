#!/usr/bin/env python3

'''

Preprocessing

'''

import itertools
import json
import logging
import re
from html.parser import HTMLParser

import pandas as pd
from autocorrect import spell
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from wordsegment import load, segment

logging.basicConfig(level=logging.INFO)
from crawling_preprocessing import expReplace as expReplace


def replaceNot(tweet):
    dText = tweet.split()
    text_list = []
    count_neg = 0
    for text_word in dText:

        if 'not' == text_word or 'Not' == text_word:
            text_word = 'negation'
            count_neg +=1
        text_list.append(text_word)
        tweet = " ".join(text_list)
   # print(tweet)

    return (tweet,count_neg)

def countEmojis(emoji_tweet):
    dText = emoji_tweet.split()
    text_list = []
    count_good = 0
    count_bad = 0
    for text_word in dText:

        if 'good' == text_word or 'smile' == text_word or 'heart' == text_word:
            count_good += 1
        if 'bad' == text_word or 'sad' == text_word or 'worry' == text_word or 'angry' == text_word:
            count_bad += 1

    # print(tweet)
    return (emoji_tweet,count_good,count_bad)


def preprocess(fp):
    '''
    Loads the dataset .txt file with label-tweet on each line and parses the dataset.
    :param fp: filepath of dataset
    :return:
        corpus: list of tweet strings of each tweet.
        y: list of labels
    '''
    #y = []
    corpus = []
    load()
    stopWords = set(stopwords.words('english'))
    #data = {'index Label' :[], 'Tweet Text': []};
    data = {'Tweet Text': [] , 'Negation Count' : [] , 'Good Emoji Count' :[] , 'Bad Emoji Count' :[] };

    with open(fp, 'rt', encoding='utf-8') as data_in:
        for line in data_in:
            if not line.lower().startswith("tweet index"): # discard first line if it contains metadata
                line = line.rstrip() # remove trailing whitespace
               # label = int(line.split("\t")[1])
                tweet = line.split("\t")[0]

                 # remove url
                tweet = re.sub(r'(http|https?|ftp)://[^\s/$.?#].[^\s]*', '', tweet, flags=re.MULTILINE)
                tweet = re.sub(r'[http?|https?]:\\/\\/[^\s/$.?#].[^\s]*', '', tweet, flags=re.MULTILINE)

                # remove mentions
                remove_mentions = re.compile(r'(?:@[\w_]+)')
                tweet = remove_mentions.sub('',tweet)

                #replace and count good , bad emojis

                emoji_tweet = expReplace.replace_emojis(tweet)
                tweet ,count_good , count_bad = countEmojis(emoji_tweet)

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

                # Standardising words
                tweet = ''.join(''.join(s)[:2] for _, s in itertools.groupby(tweet))

                # contractions applied
                words = tweet.split()
                tweet = [apoDict[word] if word in apoDict else word for word in words]
                tweet = " ".join(tweet)

                hashWords =  re.findall(r'(?:^|\s)#{1}(\w+)', tweet)
                # replace #word with word
                tweet = re.sub(r'(?:^|\s)#{1}(\w+)', r' \1', tweet)

                # word segmentation
                token_list =  word_tokenize(tweet)
                segmented_word = []
                for i in token_list:
                    if i in hashWords:
                        seg = segment(i)
                        segmented_word.extend(seg)
                    else:
                        segmented_word.append(i)

                tweet = ' '.join(segmented_word)

                # remove special symbols
                tweet = re.sub('[@#$._|]', ' ', tweet)

                # remove extra whitespaces
                tweet = re.sub('[\s]+', ' ', tweet)

                # Spelling correction
                spell_list = word_tokenize(tweet)
                filterlist = []
                for i in spell_list:
                    correct_spell = spell(i)
                    filterlist.append(correct_spell)
                tweet = ' '.join(filterlist)

                # lemma
                tweet = word_tokenize(tweet)
                lemma_tweet = []
                for tweet_word in tweet:
                    lemma_tweet.append(WordNetLemmatizer().lemmatize(tweet_word,'v'))

                tweet = ' '.join(lemma_tweet)

                # replace not with negation .. function call

                tweet,count_neg = replaceNot(tweet)
                print(count_neg)
                # remove stop words

                token_list = word_tokenize(tweet)
                wordsFiltered = []
                for i in token_list:
                    if i not in stopWords:
                        wordsFiltered.append(i)
                tweet = ' '.join(wordsFiltered)

                # remove open or empty lines
                if not re.match(r'^\s*$', tweet):
                    if not len(tweet) <= 3:
                        corpus.append(tweet)
                        #y.append(label)
                        print(tweet)

                #creating a dictionary Good Emoji Count' :[] , 'Bad Emoji Count'
                data['Tweet Text'].append(tweet)
                data['Negation Count'].append(count_neg)
                data['Good Emoji Count'].append(count_good)
                data['Bad Emoji Count'].append(count_bad)
               # data['index Label'].append(label)
    #creating a dataframe
    df = pd.DataFrame(data)
    #to order the columns in the csv file (while copying from dataframes to a CSV file)
    df_reorder = df[['Tweet Text','Negation Count','Good Emoji Count','Bad Emoji Count']]
    #writing dataframe to a csv file
    #df_reorder.to_csv('semeval_preprocessed_training_data.csv', encoding='utf-8', index= False)
    df_reorder.to_csv('/home/stark/PycharmProjects/untitled/AgainstTweetsWorld_PreProcessed.csv', encoding='utf-8', index= False)

    return corpus  #, y

# Contractions
def loadAppostophesDict(fp):
    apoDict = json.load(open(fp))
    return apoDict

if __name__ == "__main__":
    DATASET_FP = "/home/stark/PycharmProjects/untitled/AgainstTweetsWorld_UniqueRec.csv"
    #DATASET_FP = "./SemEval2018-T3-train-taskA.txt"
    APPOSTOPHES_FP = "appos.txt"
    apoDict = loadAppostophesDict(APPOSTOPHES_FP)
    #corpus, y = preprocess(DATASET_FP)
    corpus = preprocess(DATASET_FP)
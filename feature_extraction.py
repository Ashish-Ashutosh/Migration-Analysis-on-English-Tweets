
import string

import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from textblob import TextBlob

import sentiment
from crawling_preprocessing import expReplace

dict_sad={":-(":"SAD", ":(":"SAD", ":-|":"SAD",  ";-(":"SAD", ";-<":"SAD", "|-{":"SAD"}
dict_happy={":-)":"HAPPY",":)":"HAPPY", ":o)":"HAPPY",":-}":"HAPPY",";-}":"HAPPY",":->":"HAPPY",";-)":"HAPPY"}

tokenizer = RegexpTokenizer(r'\w+')
sentiments = sentiment.tweetSentiment()

''' Punctuation and special symbol features '''

''' 1.Counting Exclamation marks '''
def ExclamationCount(features, tweet):
    count=0
    threshold=2
    for word in range(len(tweet)):
        count+= int(tweet[word]=='!')
    features['exclamation'] = int(count>=threshold)


'''Sentiment score feature of the tweet '''
def getTweetSentiment(features, tweet):
    tweetSentiment= expReplace.replace_emojis(tweet)
    tokenized_tweet = tokenizer.tokenize(tweetSentiment)
    tokenized_tweet = [(t.lower()) for t in tokenized_tweet]

    tSentiment = sentiments.TweetScore(tokenized_tweet)
    features['Positive sentiment'] = tSentiment[0]
    features['Negative sentiment'] = tSentiment[1]
    features['Sentiment'] = tSentiment[0] - tSentiment[1]

    try:
        blob = TextBlob(
            "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in tokenized_tweet]).strip())
        features['Blob sentiment'] = blob.sentiment.polarity
        features['Blob subjectivity'] = blob.sentiment.subjectivity
    except:
        features['Blob sentiment'] = 0.0
        features['Blob subjectivity'] = 0.0

    '''1. sentiment score of 2 parts of the tweet and the difference between both'''
    if len(tokenized_tweet)==1:
        tokenized_tweet+=['.']
    NoOfWords = int(len(tokenized_tweet)/2)

    firstHalf = tokenized_tweet[0:NoOfWords]
    secondHalf = tokenized_tweet[NoOfWords:]

    first_tSentiment = sentiments.TweetScore(firstHalf)
    features['First Positive sentiment'] = first_tSentiment[0]
    features['First Negative sentiment'] = first_tSentiment[1]
    features['First Part Sentiment'] = first_tSentiment[0] - first_tSentiment[1]

    second_tSentiment = sentiments.TweetScore(secondHalf)
    features['Second Positive sentiment'] = second_tSentiment[0]
    features['Second Negative sentiment'] = second_tSentiment[1]
    features['Second Part Sentiment'] = second_tSentiment[0] - second_tSentiment[1]

    features['Sentiment Contrast'] = np.abs(features['First Part Sentiment'] - features['Second Part Sentiment'])

    #TextBlob Sentiment analysis
    try:
        textSentiment = TextBlob("".join([" "+ word if not word.startswith("'") and word not in string.punctuation else word for word in firstHalf]).strip())
        features['first part blob sentiment'] = textSentiment.sentiment.polarity
        features['firs part blob subjective'] = textSentiment.sentiment.subjectivity
    except:
        features['first part blob sentiment']= 0.0
        features['first part blob subjective'] = 0.0

    try:
        textSentiment = TextBlob("".join([" "+word if not word.startswith("'") and word not in string.punctuation else word for word in secondHalf]).strip())
        features['second part blob sentiment'] = textSentiment.sentiment.polarity
        features['second part blob subjective'] = textSentiment.sentiment.subjectivity
    except:
        features['second part blob sentiment'] = 0.0
        features['second part blob subjective'] = 0.0

    features['Half Sentiment blob Contrast']= np.abs(features['first part blob sentiment'] - features['second part blob sentiment'])


    '''2. sentiment score of 3 parts of the tweet and their contrast'''
    if len(tokenized_tweet)==2:
        tokenized_tweet+=['.']
    firstPart = tokenized_tweet[0:int(len(tokenized_tweet)/3)]
    secondPart = tokenized_tweet[int(len(tokenized_tweet)/3):int(2*len(tokenized_tweet)/3)]
    thirdPart = tokenized_tweet[int(2*len(tokenized_tweet)/3):]

    firstThird_tSentiment = sentiments.TweetScore(firstPart)
    features['First-Third Positive sentiment'] = firstThird_tSentiment[0]
    features['First-Third Negative sentiment'] = firstThird_tSentiment[1]
    features['First-Third Part Sentiment'] = firstThird_tSentiment[0] - firstThird_tSentiment[1]

    secondThird_tSentiment = sentiments.TweetScore(secondHalf)
    features['Second-Third Positive sentiment'] = tSentiment[0]
    features['Second-Third Negative sentiment'] = tSentiment[1]
    features['Second-Third Part Sentiment'] = secondThird_tSentiment[0] - secondThird_tSentiment[1]

    third_tSentiment = sentiments.TweetScore(secondHalf)
    features['Third Positive sentiment'] = third_tSentiment[0]
    features['Third Negative sentiment'] = third_tSentiment[1]
    features['Third Part Sentiment'] = third_tSentiment[0] - third_tSentiment[1]

    features['3 Sentiment contrast'] = np.abs(features['First-Third Part Sentiment'] - features['Third Part Sentiment'])

    '''features['firstsecond sentiment Contrast'] = np.abs(features['First-Third Part Sentiment'] - features['Second-Third Part Sentiment'])'''
    features['firstthird sentiment Contrast'] = np.abs(features['First-Third Part Sentiment'] - features['Third Part Sentiment'])
    '''features['secondthird sentiment Contrast'] = np.abs(features['Second-Third Part Sentiment'] - features['Third Part Sentiment'])'''


    try:
        text = TextBlob("".join([" "+word if not word.startswith("'") and word not in string.punctuation else word for word in firstPart]).strip())

        features['first-third part sentiment blob'] = text.sentiment.polarity
        features['first-third part subjective blob'] = text.sentiment.subjectivity

    except:
        features['first-third part sentiment blob'] = 0.0
        features['first-third part subjective blob'] = 0.0

    try:
        text = TextBlob("".join([" " + word if not word.startswith("'") and word not in string.punctuation else word for word in secondPart]).strip())

        features['second-third part sentiment blob'] = text.sentiment.polarity
        features['second-third part subjective blob'] = text.sentiment.subjectivity

    except:
        features['second-third part sentiment blob'] = 0.0
        features['second-third part subjective blob'] = 0.0

    try:
        text = TextBlob("".join([" "+word if word not in string.punctuation and not word.startswith("'") else word for word in thirdPart]).strip())

        features['third part sentiment blob'] = text.sentiment.polarity
        features['third part subjective blob'] = text.sentiment.subjectivity

    except:
        features['third part sentiment blob'] = 0.0
        features['third part subjective blob'] = 0.0

    '''features['firstsecond sentiment blob Contrast'] = np.abs(features['first-third part sentiment blob'] - features['second-third part sentiment blob'])'''
    features['firstthird sentiment blob Contrast'] = np.abs(features['first-third part sentiment blob'] - features['third part sentiment blob'])
    '''features['secondthird sentiment blob Contrast'] = np.abs(features['second-third part sentiment blob'] - features['third part sentiment blob'])'''


'''2. UpperCase'''
def UpperCaseFeature(features, tweet):
    count=0
    threshold=1
    tokenized_tweet = nltk.word_tokenize(str(tweet))
    for word in tokenized_tweet:
        count += int(word.isupper())
    features['UpperCase'] = int(count>=threshold)


def findNegation(features, tweet):
	countNegation = 0
	for i in range(len(tweet)):
		if tweet[i]=='negation':
			countNegation+=1
	features['Negation']=countNegation


'''Part-Of-Speech tagging'''
def getPOSfeature(features, tweet):
    posTweet = expReplace.replace_emojis(tweet)
    tokens = tokenizer.tokenize(posTweet)
    tokens = [(t.lower()) for t in tokens]
    pos_vector= sentiments.positionVector(tokens)

    for j in range(len(pos_vector)):
        features['POS' + str(j+1)] = pos_vector[j]


'''Bigrams'''
def getbigramsfeatures(features, tweet):
    tokens= tokenizer.tokenize(tweet)
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(word) for word in tokens]
    bigrams = nltk.bigrams(lemmas)
    bigrams = [part[0]+' '+part[1] for part in bigrams]
    bigramfeat = lemmas + bigrams

    for feat in bigramfeat:
        features['contains(%s)' %feat] = 1.0

def count_emotion(features,sentence):
    returnsent = sentence
    happy = 0;
    sad = 0
    for i in dict_happy:
        happy += returnsent.count(i)
    for i in dict_sad:
        sad += returnsent.count(i)
    features['happyemo'] = happy
    features['sademo'] = sad

def getallfeatureset(tweet):
    features = {}
    UpperCaseFeature(features,tweet)
    getTweetSentiment(features,tweet)
    ExclamationCount(features,tweet)
    getbigramsfeatures(features,tweet)
    getPOSfeature(features,tweet)
    count_emotion(features, tweet)
    return features

print(getallfeatureset("In favour of migrants"))
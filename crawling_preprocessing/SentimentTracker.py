import string

import pandas as pd
from nltk.tokenize import RegexpTokenizer
from textblob import TextBlob

import sentiment as sentiment
from crawling_preprocessing import expReplace as expReplace

tokenizer = RegexpTokenizer(r'\w+')
sentiments = sentiment.tweetSentiment()

'''Sentiment score feature of the tweet '''
def getTweetSentiment(features, tweet):
   tweetSentiment= expReplace.replace_emojis(tweet)
   tokenized_tweet = tokenizer.tokenize(tweetSentiment)
   tokenized_tweet = [(t.lower()) for t in tokenized_tweet]

   tSentiment = sentiments.TweetScore(tokenized_tweet)
   features['Sentiment'] = tSentiment[0] - tSentiment[1]

   try:
       blob = TextBlob(
           "".join([" " + i if not i.startswith("'") and i not in string.punctuation else i for i in tokenized_tweet]).strip())
       features['Blob Sentiment'] = blob.sentiment.polarity
   except:
       features['Blob Sentiment'] = 0.0

# dictionary


def sentimentScore(tweet):
   features = {}
   getTweetSentiment(features, tweet)

   if features['Sentiment'] > 0 and features['Blob Sentiment'] > 0:
   #write label and the tweet in the new file
       return "1"
                #print(1)
   elif features['Sentiment'] < 0 and features['Blob Sentiment'] < 0:
       return "-1"
                #print(-1)
   elif features['Blob Sentiment']==0.0:
       return "0"
                #print(0)

label_list = []

if __name__ == "__main__":

    df = pd.read_csv('/home/stark/PycharmProjects/untitled/AgainstTweetsWorld_PreProcessed.csv')

    for index, row in df.iterrows():
        tweet = row['Tweet Text'].split("\t")

        print(tweet[0])
        value = sentimentScore(tweet[0])
        label_list.append(value)

    print(len(label_list))
    df['Label'] = label_list
    df.to_csv('/home/stark/PycharmProjects/untitled/AgainstTweetsWorld_Polarity.csv', index=False)
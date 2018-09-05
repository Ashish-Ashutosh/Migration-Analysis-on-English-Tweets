""" This class is used to extract the sentiment score of a sentence.
The constructor builds a sentiment score dictionnary using SentiWordNet_3.0.0_20130122.txt, and the
function 'score' of the class uses the dictionnary to extract a positive and negative sentiment score from a word
and its part of speech by looking up the word in the dictionnary.
"""

import csv, collections, os
import numpy as np
import nltk

#classifitcation of positive and negative score, using SentiWordNet
class tweetSentiment(object):
    def __init__(self):
        sentimentScores = collections.defaultdict(list)
        #Returns a new dictionary-like object. Group a sequence of key-value pairs into a dictionary of lists

        with open(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'SentiWordNet_3.0.0_20130122.txt'), 'r') as csvfile:

            reader = csv.reader(csvfile, delimiter='\t', quotechar='"')

            for line in reader:
                if line[0].startswith('#'):
                    continue
                if len(line) == 1:
                    continue

                POS, ID, PosScore, NegScore, SynsetTerms, Gloss = line

                if len(POS) == 0 or len(ID) == 0:
                    continue

                for term in SynsetTerms.split(" "):
                    term = term.split('#')[0]
                    term = term.replace("-", " ").replace("_", " ")
                    key = "%s/%s" % (POS, term.split("#")[0])
                    sentimentScores[key].append((float(PosScore), float(NegScore))) #returns the list of scores for that key [('key', [posscore, negscore]), ...]
                    #we classify all the words by + or -

        #Returns the average of the array elements(scores)
        for key, value in sentimentScores.items():
            sentimentScores[key]= np.mean(value, axis=0)

        self.sentimentScores = sentimentScores


    def WordScore(self, word):
        pos = nltk.pos_tag([word])[0][1]
        return self.score(word, pos)

    def TweetScore(self, tweet):
        pos = nltk.pos_tag(tweet)
        scores = np.array([0.0, 0.0])
        for j in range(len(pos)):
            scores +=self.score(pos[j][0], pos[j][1])

        return scores

    def score(self, word, pos):
        if pos[0:2] == 'NN': #noun
            pos_type = 'n'
        elif pos[0:2] == 'JJ': #adjective
            pos_type = 'a'
        elif pos[0:2] == 'VB': #verb
            pos_type = 'v'
        elif pos[0:2] == 'RB':  # adverb
            pos_type = 'r'
        else:
            pos_type = 0

        if pos_type!=0:
            dictionaryLocation = pos_type + '/' + word
            posNegScores = self.sentimentScores[dictionaryLocation]
            if len(posNegScores) == 2:
                return posNegScores
            else:
                return np.array([0.0, 0.0])
        else:
            return np.array([0.0, 0.0])


    def positionVector(self, tweet):
        pos_vector = nltk.pos_tag(tweet)
        vector = np.zeros(4)

        for j in range(len(tweet)):
            pos = pos_vector[j][1]
            if pos[0:2] == 'NN':
                vector[0] += 1
            elif pos[0:2] == 'JJ':
                vector[1] += 1
            elif pos[0:2] == 'VB':
                vector[2] += 1
            elif pos[0:2] == 'RB':
                vector[3] += 1
        return vector

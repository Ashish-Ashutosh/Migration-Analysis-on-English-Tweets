import numpy as np
import pickle
import os
import feature_extraction
import warnings

#dictionary vector object to convert from the lists of feature value mappings to vectors for training
vecFile = open('dictionaryFileLR.p', 'rb')
#classifier which is trained by the training.py
classFile = open('classifierLR.p', 'rb')


with warnings.catch_warnings():
      warnings.simplefilter("ignore", category=UserWarning)
      vector = pickle.load(vecFile)
      classifier = pickle.load(classFile)

vecFile.close()
classFile.close()

#Gives the percentage of the ironic score of a tweet
def getIronicScore(tweet):
    features = feature_extraction.getallfeatureset(tweet)
    # classifier can only get data in numerical form so we convert it in vector form.
    featuresVector = vector.transform(features)
    #	Distance of the samples featureVector to the separating hyperplane
    score = classifier.decision_function(featuresVector)[0]
    #sigmoid
    percentage = int(round(2.0*(1.0/(1.0+np.exp(-score))-0.5)*100.0))

    return percentage

while True:
    print("Enter the tweet to get the ironic score or type exit to quit")
    data = input()
    if data=="exit":
        break;
    else:
        print(getIronicScore(data))
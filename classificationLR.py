""" This function is used to do experiments on the data and to save a dictionnary of features
and a classifier. It first generates the features for all tweets and then train a classifier
using these features.  It outputs a report showing the accuracy of the classifier. """

import scipy as sp
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction import DictVectorizer
import pickle
import feature_extraction
from sklearn.linear_model import LogisticRegression
import itertools
import numpy as np
import matplotlib.pyplot as plt


print('Pickling out')
positive_data = np.genfromtxt('Data/FavourUni.csv', skip_header=True, delimiter=";", dtype=None , encoding= None)
negative_data = np.genfromtxt('Data/AgainstUni.csv', skip_header=True, delimiter=";", dtype=None , encoding= None)
#positive_data = np.load('positive.npy')
#negative_data = np.load('neg.npy')
#neutral_data = np.load('ironic_semeval_test3.npy')

#print('Number of ironic tweets :', len(positive_data))
#print('Number of ironic tweets :', len(negative_data))

print('Feature engineering')
classificationSet = ['Positive', 'Negative'] #label set
featureSets = []

index=0
for tweet in positive_data:
    if (np.mod(index, 1000) == 0):
        print("Processed Positive Tweets: ", index)
    featureSets.append((feature_extraction.getallfeatureset(tweet), classificationSet[0]))
    index+=1

index = 0
for tweet in negative_data:
    if (np.mod(index, 1000) == 0):
        print("Processed Negative Tweets: ", index)
    featureSets.append((feature_extraction.getallfeatureset(tweet), classificationSet[1]))
    index+=1

featureSets=np.array(featureSets)
targets=(featureSets[0::,1]=='Negative').astype(int)

#Transforms lists of feature-value mappings to vectors
vector = DictVectorizer()
featureVector = vector.fit_transform(featureSets[0::,0])

#Saving the dictionary vectorizer
fName = "outputfiles/dictionaryFileLR.p"
#write binary mode
fObject = open(fName, 'wb')
pickle.dump(vector, fObject)
fObject.close()

#Feature splitting
print('Feaature Splitting')
order = shuffle(list(range(len(featureSets))))

targets=targets[order]
featureVector=featureVector[order, 0::]

#Splitting data set in training and test set
size = int(len(featureSets) * .3)

trainVector = featureVector[size:,0::]
trainTargets = targets[size:]
testVector = featureVector[:size, 0::]
testTargets = targets[:size]

print('Training')

#Artificial weights
negativeP=(trainTargets==1)
positiveP = (trainTargets==0)
ratio = np.sum(negativeP.astype(float))/np.sum(positiveP.astype(float))

newTrainVector = trainVector
newTrainTargets=trainTargets

for j in range(int(ratio-1.0)):
    newTrainVector=sp.sparse.vstack([newTrainVector,trainVector[positiveP,0::]]) #Stack sparse matrices vertically
    newTrainTargets=np.concatenate((newTrainTargets,trainTargets[positiveP]))

#classifier = SVC(C=0.1, kernel='linear')
classifier =  LogisticRegression()
classifier.fit(newTrainVector, newTrainTargets)

#Saving the classifier
fName = "outputfiles/classifierLR.p"
fObject=open(fName,'wb')
pickle.dump(classifier,fObject)
fObject.close()


#Validation
print('Validating')
class_names = ['positive', 'negative']
output = classifier.predict(testVector)
classificationReport = classification_report(testTargets,output,target_names=classificationSet)
print(classificationReport)
print(accuracy_score(testTargets, output)*100)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


#compute confusion matrix
cnf_matrix=confusion_matrix(testTargets, output)
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names,
                      title='Confusion matrix, without normalization LR')

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix LR')

plt.show()


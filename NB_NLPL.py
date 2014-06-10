import csv
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.corpus import stopwords
import random
import math
import time
 
def word_feats(words):
    return dict([(word, True) for word in words])

def randomList(a):
    b =[]
    for i in range(len(a)):
        element = random.choice(a)
        a.remove(element)
        b.append(element)
        return b

def category(a):
    return {
        '1' : 'Negative',
        '2' : 'S Negative',
        '3' : 'Neutral',
        '4' : 'S Positive',
        '5' : 'Positive' }.get(a)

#Build a training data set

stop = stopwords.words('english')
with open("train.tsv") as csvfile:
    records = csv.reader(csvfile, delimiter='\t')
    next(records)
    t = [({word: True for word in nltk.word_tokenize(row[2]) if word not in stop}, category(row[3]))for row in records]
        #t = [({word: True for word in nltk.word_tokenize(row[2])}, category(row[3]))for row in records]
    
trainlen = int((len(t) * 3 / 4))
train = t[:trainlen]
test = t[trainlen:]
    

###test file data for later.  Might want to incorporate a database read
##with open("train.tsv") as csvfile:
##    records = csv.reader(csvfile, delimiter='\t')
##    test = [({word: True for word in nltk.word_tokenize(row[2])}, row[3])for row in records]
##

#classifier NaiveBayes
timer = time.clock()
print('NaiveBayes Model')
classifier = NaiveBayesClassifier.train(train)
print ('accuracy NaiveBayes:', nltk.classify.util.accuracy(classifier, test))
classifier.show_most_informative_features(20)
print('NaiveBayes Time: ' + str(time.clock() - timer))
print('/n')


#classifier DecisionTree
timer = time.clock()
print('Decision Tree Model')
classifier2 = nltk.DecisionTreeClassifier.train(train)
print ('accuracy DecisionTree:', nltk.classify.util.accuracy(classifier2, test))
print('DecisionTree Time: ' + str(time.clock() - timer))






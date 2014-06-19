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
    t = [({word: True for word in nltk.word_tokenize(row[2]) if word not in stop}, (row[3]))for row in records]
print('Train record count: ' + str(len(t)))    
##trainlen = int((len(t) * 3 / 4))
##train = t[:trainlen]
##test = t[trainlen:]

##test file data for later.  Might want to incorporate a database read
with open("test.tsv") as csvfile:
    records2 = csv.reader(csvfile, delimiter='\t')
    next(records2)
    test2 = [({word: True for word in nltk.word_tokenize(row[2]) if word not in stop}, row[0])for row in records2]
print('Test record count: ' + str(len(test2)))

#classifier NaiveBayes
##timer = time.clock()
##print('NaiveBayes Model')
##classifier = NaiveBayesClassifier.train(train)
##print ('accuracy NaiveBayes:', nltk.classify.util.accuracy(classifier, test))
##classifier.show_most_informative_features(20)
##print('NaiveBayes Time: ' + str(time.clock() - timer))
##print('/n')

###classifier DecisionTree
##timer = time.clock()
##print('Decision Tree Model')
##classifier = nltk.DecisionTreeClassifier.train(t)
###print ('accuracy DecisionTree:', nltk.classify.util.accuracy(classifier, test))
##print('DecisionTree Time: ' + str(time.clock() - timer))

#classifier SVM
timer = time.clock()
print('SVM Model')
classifier = nltk.classify.scikitlearn.SklearnClassifier.train(t)
#print ('accuracy DecisionTree:', nltk.classify.util.accuracy(classifier, test))
print('SVM Time: ' + str(time.clock() - timer))

outfile = open('out_DecisionTree.txt','w')
outfile.write('PhraseId,Sentiment' + '\n')
for i in test2:
    a, b = i
    c = classifier.classify(a)
    if c == '0':
        c = 3
    else:
        c = c
    outfile.write(str(b) +','+ str(c)+'\n')
    
outfile.close()




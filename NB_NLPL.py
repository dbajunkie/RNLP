import csv
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
 
def word_feats(words):
    return dict([(word, True) for word in words])

object = []
with open("train2.tsv") as csvfile:
    records = csv.reader(csvfile, delimiter='\t')
    for row in records:
        rowx = nltk.word_tokenize(row[2])
        mydict = {k: True for k in rowx}
        object.append(mydict)
        mydict = {}
    print(object)




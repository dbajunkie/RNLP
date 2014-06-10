import csv
import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize # or use some other tokenizer

with open("train2.tsv") as csvfile:
    records = csv.reader(csvfile, delimiter='\t')
    all_words = set(word.lower() for corp in records for word in word_tokenize(corp[2]))
    print(all_words)
    t = [({word: (word in word_tokenize(x[2])) for word in all_words}, x[2]) for x in records]
print(t)

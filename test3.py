import nltk
alpha = ['a','b','c']
beta = ['x','y']

train = [(dict(a=1,b=1,c=1), 'y')]
print(train)

classifier = nltk.classify.NaiveBayesClassifier.train(train)
sorted(classifier.labels())

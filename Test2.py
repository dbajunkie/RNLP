

from nltk.corpus import names
def gender_features(word):
    return {'last_letter': word[-1]}
gender_features('Shrek')



import random
names = ([(name, 'male') for name in names.words('male.txt')] +
         [(name, 'female') for name in names.words('female.txt')])
print(names[:2])

featuresets = [(gender_features(n), g) for (n,g) in names]
print(featuresets[:2])
#print(featuresets)

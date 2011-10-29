import random

"""
A classifier is initialized with a training set of feature vectors
and has a classify method that takes in a new vector and returns
a class
"""

class OneClassifier:
    def __init__(self, trainingset) :
        self.trainingset = trainingset
        self.classes = (True, False)
    def classify(self, point):
        return 1

class RandomClassifier:
    def __init__(self, trainingset) :
        self.trainingset = trainingset
        self.classes = (True, False)
    def classify(self, point):
        return random.randint(0,1)

class BayesClassifier:
    def __init__(self, trainingset) :
        self.cl = {}
        for row in trainingset.T:
            if row[-1] in self.cl:
                v, c = cl[row[-1]]
                self.cl[row[-1]] = (v + row[:-1], c + sum(row[:-1]))
            else:
                self.cl[row[-1]] = (row[:-1], sum(row[:-1]))
    def classify(self, point):
        for c in self.cl:
            
            

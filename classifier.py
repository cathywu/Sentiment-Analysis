import random
import data
from numpy import *

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
        self.classes = {}
        self.lengths = {}
        for row in trainingset.asMatrix().T:
            cls = row[-1]
            pt = row[:-1]
            if cls in self.classes:
                self.classes[cls] += pt - ones(len(pt))

            else:
                self.classes[cls] = pt

        for cls in self.classes:
            self.lengths[cls] = sqrt(float(dot(self.classes[cls], self.classes[cls])))
            self.classes[cls] = log(self.classes[cls])
    def classify(self, point):
        mx = 0
        mx_cls = 0
        for cls in self.classes:
            dotprod = dot(self.classes[cls], log(point)) - log(self.lengths[cls])
            if dotprod > mx:
                mx = dotprod
                mx_cls = cls
        return mx_cls
                
class BayesPresenceClassifier(BayesClassifier):
    def classify(self, point):
        mx = 0
        mx_cls = 0
        for cls in self.classes:
            dotprod = dot(self.classes[cls], log(point.clip(max=2))) - log(self.lengths[cls])
            if dotprod > mx:
                mx = dotprod
                mx_cls = cls
        return mx_cls
    
    
def test_bayes():
    trainingset = data.Data(array([[2, 2, 2, 1],
                                   [1, 1, 2, 0],
                                   [1, 1, 2, 0],
                                   [2, 1, 1, 0]]).T)
    bc = BayesClassifier(trainingset)
    print bc.classify(array([2, 2, 2]))
    print bc.classify(array([3, 1, 1]))

if __name__ == "__main__":
    test_bayes()

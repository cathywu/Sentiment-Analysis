#!/usr/bin/python

import random
import data
from numpy import *
from PyML import *
from PyML.containers import *
from scipy.sparse import csr_matrix, lil_matrix, csc_matrix, issparse
import sys
from ngrams import *
import tempfile

"""
A classifier has a addFeatureVector method that takes a feature
vector, which is a dictionary that maps feature names to values,
like {"word1":4, "word2":2, ...} and has a classify method that 
takes in a new vector and returns a class
"""

class Classifier:
    def __init__(self):
        self.nfeatures = 0
        self.nvectors  = 0
        self.index     = {}

    def addToIndex(self, words):
        self.compiled = False
        words = set([i for i in words])
        keys = set(self.index.keys())
        words = words - keys
        for w in words:
            self.index[w] = self.nfeatures
            self.nfeatures += 1
            
    def vectorFromDict (self, words):
        self.addToIndex(words.keys())
        vec = zeros(self.nfeatures)
        for w in words:
            vec[self.index[w]] = words[w]
        return vec
                
            
class OneClassifier:
    def addFeatureVector (self, vec, cls): 
        pass
    def classify(self, point):
        return 1


class RandomClassifier:

    def addFeatureVector (self, vec, cls):
        pass
    def classify(self, point):
        return random.randint(0,1)


class BayesClassifier(Classifier):
    def __init__(self) :
        Classifier.__init__(self)
        self.length    = 0
        self.compiled  = True
        self.classes   = {}
    def addToIndex(self, words):
        words = set(words) - set(self.index.keys())
        for cls in self.classes:
            self.classes[cls] = hstack((self.classes[cls], ones(len(words))))
        Classifier.addToIndex(self, words)
        
    def addFeatureVector(self, vec, cls):
        self.compiled = False
        if cls not in self.classes:
            self.classes[cls] = ones(self.nfeatures)
        self.addToIndex(vec)
        for feature in vec:
            self.classes[cls][self.index[feature]] += vec[feature]
        self.nvectors += 1
        self.length += 1;
    def compile(self):
        if self.compiled:
            return
        self.compiled = True
        self.normalized = self.classes
        self.lengths = {}
        for i in range(self.nfeatures):
            total = 0
            for cls in self.classes:
                total += self.classes[cls][i]
            for cls in self.classes:
                self.normalized[cls][i] = float(self.classes[cls][i])/total
        for cls in self.classes:
            self.lengths[cls] = 0
            for i in range(self.nfeatures):
                self.lengths[cls] += self.classes[cls][i]
            self.lengths[cls] = sqrt(self.lengths[cls])
            for i in range(self.nfeatures):
                self.normalized[cls][i] /= self.lengths[cls]

    def classify(self, vec):
        self.compile()
        mx = -sys.maxint
        mx_cls = 0
        point = ones(self.nfeatures)

        for feature in vec:
            if feature in self.index:
                point[self.index[feature]] += vec[feature]
        for cls in self.classes:
            dotprod = dot(log(self.classes[cls]), log(point)) - log(self.lengths[cls])
            if dotprod > mx:
                mx = dotprod
                mx_cls = cls
        return mx_cls


class BayesPresenceClassifier(BayesClassifier):
    def classify(self, point):
        return BayesClassifier.classify(self, point.clip(max=2))


class LinearSVMClassifier(Classifier):
    def __init__(self):
        Classifier.__init__(self)
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.filename = self.file.name
        print self.filename
        self.data = SparseDataSet(0)
        self.svm = SVM(optimizer='liblinear')

    def vectorToString(self, vec, cls):
        return str(cls) + " " + " ".join([str(i) + ":" + str(vec[i])for i in vec]) + "\n"

    def addFeatureVector(self, point, cls):
        self.compiled = False
        vec = self.vectorToString(point, cls)
        self.file.write(vec)
        
    def compile(self):
        if self.compiled == True:
            return
        self.compiled = True
        self.file.close()
        self.data = SparseDataSet(self.filename)
#        self.svm.train(self.data)
        self.file = open(self.filename)

    def validate(self, n):
#        self.compile()
#        v = self.vectorFromDict(point)
        
#        outp = self.svm.test(v)
        self.compile()
        print self.data
        outp = self.svm.cv(self.data, numFolds = n)
        print outp

        
        
def test_bayes():
    trainingset = array([[2, 2, 2, 1],
                         [1, 1, 2, 0],
                         [1, 1, 2, 0],
                         [2, 1, 1, 0]]).T
    bc = BayesClassifier()
    for vec in trainingset:
        bc.addFeatureVector(vec[:-1], vec[-1])
    print bc.classify(array([2, 2, 2]))
    print bc.classify(array([3, 1, 1]))

            
def test_svm():
    trainingset = [ngrams(1, "foo foo bar baz"), ngrams(1, "foo foo bar bar baz baz"), ngrams(1,"foo foo bar baz")]
    labels = [1, -1, -1]
    lsc = LinearSVMClassifier(3)
    for vec in zip(trainingset, labels):
        lsc.addFeatureVector(vec[0], vec[1])
    print lsc.classify(ngrams(1, "foo foo bar bar baz baz"))
    print lsc.classify(ngrams(1, "foo foo foo bar baz"))

if __name__ == "__main__":
    test_svm()

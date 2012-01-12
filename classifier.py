#!/usr/bin/python

import random
import data
from numpy import *
from PyML import *
from PyML.containers import *
from scipy.sparse import csr_matrix, lil_matrix, csc_matrix, issparse
import sys

"""
A classifier has a addFeatureVector method that takes a feature
vector, which is a dictionary that maps feature names to values,
like {"word1":4, "word2":2, ...} and has a classify method that 
takes in a new vector and returns a class
"""

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


class BayesClassifier:
    def __init__(self) :
        self.classes = {}
        self.nfeatures = 0
        self.nvectors = 0
        self.index   = {}
        self.length = 0

    def addToIndex(self, words):
        words = set([i for i in words])
        keys = set(self.index.keys())
        words = words - keys
        
        for w in words:
            self.index[w] = self.nfeatures
            self.nfeatures += 1
        for cls in self.classes:
            self.classes[cls] = hstack((self.classes[cls], ones(len(words))))
    def addFeatureVector(self, vec, cls):

        if cls not in self.classes:
            self.classes[cls] = ones(self.nfeatures)
            
        for feature in vec:
            if feature not in self.index:
                for cls in self.classes:                    
                    self.classes[cls] = hstack((self.classes[cls], array([1])))
                self.index[feature] = self.nfeatures
                self.nfeatures += 1
            self.classes[cls][self.index[feature]] += vec[feature]
        self.nvectors += 1
        self.length += 1;
    def compile(self):
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


class LinearSVMClassifier:
    def __init__(self, trainingset):
        print "LinearSVM: Creating dataset"
        L = [i for i in trainingset.asMatrix().T[-1]]
        print "> L"
        X = trainingset.asMatrix().T[:-1].T
        print "> X"
        data = SparseDataSet(X.tolist(), L=L) 
        print "> data"
        self.svm = svm.SVM()
        print "Training SVM"
        self.svm.train(data)
        
    def classify(self, point):
        L= array(['1.0', '0.0'])
        X = SparseDataSet(array([point], dtype=uint16).tolist())
        print "LinearSVM: Classifying"
        return self.svm.classify(X, 0)[0]
        
        
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
    trainingset = data.Data(array([[2, 2, 2],
                                   [1, 1, 2],
                                   [1, 1, 2],
                                   [0, 1, 0]], dtype=uint16).T)
    bc = LinearSVMClassifier(trainingset)
    print bc.classify(array([2, 2, 2], dtype=uint16))
    print bc.classify(array([3, 1, 1], dtype=uint16))

if __name__ == "__main__":
    test_svm()

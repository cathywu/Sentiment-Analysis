#!/usr/bin/python

import random
import data
from numpy import *
from PyML import *
from PyML.containers import *
from maxent import MaxentModel
from scipy.sparse import csr_matrix, lil_matrix, csc_matrix, issparse
import sys
from ngrams import *
import tempfile
import os

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
    def __init__(self, restrictFeatures = False) :
        Classifier.__init__(self)
        self.length    = 0
        self.compiled  = True
        self.classes   = {}
        self.restrictFeatures = restrictFeatures
        if restrictFeatures:
            self.addToIndex(self.restrictFeatures)

    def addToIndex(self, words):
        words = set(words) - set(self.index.keys())
        for cls in self.classes:
            self.classes[cls] = hstack((self.classes[cls], ones(len(words))))
        Classifier.addToIndex(self, words)
        
    def addFeatureVector(self, vec, cls, binary=False):
        self.compiled = False
        if cls not in self.classes:
            self.classes[cls] = ones(self.nfeatures)
        if not self.restrictFeatures:
            self.addToIndex(vec)
        for feature in vec:

            if self.restrictFeatures and feature not in self.restrictFeatures:
                continue
            if feature in self.index:

                if binary:
                    self.classes[cls][self.index[feature]] += 1
                else:
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
    def __init__(self, restrictFeatures=False):
        Classifier.__init__(self)
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.filename = self.file.name
        print self.filename
        self.data = SparseDataSet(0)
        self.svm = SVM(optimizer='liblinear')
        self.restrictFeatures = restrictFeatures

    def vectorToString(self, vec, cls = False):
        # granted, this is kind of silly
        # creates a string of the format "[class if point is labeled] feature1:value1 feature2:value2..."
        # where the only allowed features are the ones in restrictFeatures, if we're restricting the features
        return ((str(cls) + " ") if cls else "") + " ".join(["-".join(str(i).split()) + ":" + str(vec[i]) 
                                                                 for i in vec if (not self.restrictFeatures) or (i in self.restrictFeatures)]) + "\n"

    def addFeatureVector(self, point, cls, binary=False):
        self.compiled = False
        vec = self.vectorToString(point, cls)
        self.file.write(vec)
        
    def compile(self):
        if self.compiled == True:
            return
        self.compiled = True
        self.file.close()
        self.data = SparseDataSet(self.filename)
        self.file = open(self.filename)
        self.svm.train(self.data)

    def validate(self, n):
        self.compile()
        print self.data
        outp = self.svm.cv(self.data, numFolds = n)
        print outp
    def classify(self, pt):
        self.compile()
        f = tempfile.NamedTemporaryFile(delete=False)
        fname = f.name
        f.write(self.vectorToString(pt))
        f.close()        
        data = SparseDataSet(fname)
        os.remove(fname)
        return int(self.svm.test(data, verbose=0).getPredictedLabels()[0])

class MaximumEntropyClassifier(Classifier):
    def __init__(self, restrictFeatures=False):
        Classifier.__init__(self)
        print "MaximumEntropy: Creating model"
        self.model = MaxentModel()
        self.model.verbose = 1
        self.restrictFeatures = restrictFeatures
        self.model.begin_add_event()

    def addToIndex(self, trainingset):
        for (vec,cls) in trainingset:
            self.addFeatureVector(vec,cls)
        
    def addFeatureVector(self, vec, cls, value=1, binary=False):
        for key in vec.keys():
            if key not in self.restrictFeatures:
                del vec[key]
        context = vec.keys()
        label = "%s" % cls
        self.model.add_event(context,label,value)

    def compile(self):
        self.model.end_add_event()
        self.model.train(30, "lbfgs", 2, 1E-03)
        #self.model.train(100, 'gis', 2)
        print "> Models trained"

    def classify(self, point, label='1'):
        result = self.model.eval(point.keys(), label)
        if result >= 0.5:
            return 1
        return -1

class MajorityVotingClassifier(Classifier):
    def __init__(self):
        self.classifiers = []
        self.reliabilities = []

    def addClassifier(self, classifier, train_files, test_files = [], reliability=1):
        self.classifiers.append(classifier)
        self.reliability.append(reliability)

    def addFeatureVector(self, vec):
        for cls in self.classifiers:
            cls.addFeatureVector(vec)

    def classify(self, vec):
        results = {}
        for cls in self.classifiers:
            r = cls.classify(vec)
            if r not in results:
                results[r] = 1
            else:
                results[r] += 1
        mx = 0
        mxarg = 0

        for r in results:
            if results[r] > mx:
                mxarg = r
                mx = results[r]
        return mxarg
        
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
    lsc = LinearSVMClassifier()
    for vec in zip(trainingset, labels):
        lsc.addFeatureVector(vec[0], vec[1])
    print lsc.classify(ngrams(1, "foo foo bar bar baz baz"))
    print lsc.classify(ngrams(1, "foo foo foo bar baz"))

def test_maxent():
    trainingset = [(['good'],'pos',1),
                   (['wonderful'],'pos',1),
                   (['ugly'],'neg',1),
                   (['terrible','ick'],'neg',1)]
    m = MaximumEntropyClassifier(trainingset)
    
    print "other label: %s" % m.classify(['mmm'],'otherlabel') # other label
    print "OOD: %s" % m.classify(['mmm'],'pos') # OOD
    print "ick: %s" % m.classify(['ick'],'pos')
    print "mmm awesome good: %s" % m.classify(['mmm','awesome','good'],'pos')
    print "mmm terrible good: %s" % m.classify(['mmm','terrible','good'],'pos')
    print "wonderful terrible good: %s" % m.classify(['wonderful','terrible','good'],'pos')

if __name__ == "__main__":
    test_maxent()

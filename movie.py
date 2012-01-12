#!/usr/bin/python

import data
import ngrams
import validate
import classifier
import os
from numpy import *


class MovieReviews:
    def __init__(self, clsf, n, testsize):
        print "Reading files"
        self.pos_files = [ngrams.ngrams(n, open("pos/"+i).read())
                          for i in os.listdir("pos")][:testsize]
        self.neg_files = [ngrams.ngrams(n, open("neg/"+i).read())
                          for i in os.listdir("neg")][:testsize]
        
        self.classifier = clsf()
        count = 0
        print "Creating Index"
        words = set()
        print "Positive index"
        words = apply(words.union, [i.keys() for i in self.pos_files])
        print "Negative index"
        words = apply(words.union, [i.keys() for i in self.neg_files])
        self.classifier.addToIndex(words)

        print "Making classifier"        
        for i in self.pos_files:
            count += 1
            self.classifier.addFeatureVector(i, 1)
        for i in self.neg_files:
            self.classifier.addFeatureVector(i, -1)
        self.classifier.compile()
        print self.classifier.classes
classif = classifier.BayesClassifier
#classif = classifier.LinearSVMClassifier
def test():
    testsize=800
    n = 1
    print "Building Classifier"
    m = MovieReviews(classif, n, testsize)
    print "Testing"
    pos_tests = [ngrams.ngrams(n, open("pos/"+i).read()) 
                      for i in os.listdir("pos")][testsize:]
    neg_tests = [ngrams.ngrams(n, open("neg/"+i).read()) 
                      for i in os.listdir("neg")][testsize:]
    pos_results = [m.classifier.classify(i) for i in pos_tests]
    print pos_results
    print len([i for i in pos_results if i == 1])
    neg_results = [m.classifier.classify(i) for i in neg_tests]
    print neg_results
    print len([i for i in neg_results if i == -1])
        
if __name__ == "__main__":
    test()

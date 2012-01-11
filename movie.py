#!/usr/bin/python

import data
import ngrams
import validate
import classifier
import os
from numpy import *


class MovieReviews:
    def __init__(self, clsf, n):
        print "Reading files"
        self.pos_files = [ngrams.ngrams(n, open("pos/"+i).read()) for i in os.listdir("pos")]
        self.neg_files = [ngrams.ngrams(n, open("neg/"+i).read()) for i in os.listdir("neg")]
        self.classes = [1] * len(self.pos_files) + [0] * len(self.neg_files)
        print "Building dictionary"
        self.dictionary = ngrams.ngrams_to_dictionary(self.pos_files + self.neg_files)
        print len(self.dictionary)
        print "Building matrix"
        self.mat = ngrams.ngrams_to_sparse(self.pos_files + self.neg_files, self.classes)
        print "Building classifier"
        self.classifier = clsf(self.mat)


classif = classifier.SparseBayesClassifier
#classif = classifier.LinearSVMClassifier
def test():
    print "Reading and parsing files..."
    m = MovieReviews(classif, 2)

    print "Creating matrix..."

    print "Running classifier..."
    print validate.kfold(3, classif, m.mat)
    print validate.kfold(5, classif, m.mat)
    print validate.kfold(10, classif, m.mat)

if __name__ == "__main__":
    test()

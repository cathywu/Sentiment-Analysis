import data
import ngrams
import validate
import classifier
import os
from numpy import *
classif = classifier.BayesPresenceClassifier

def read_reviews():
    print "Reading and parsing files..."
    pos_files = [ngrams.ngrams(1, open("pos/"+i).read()) for i in os.listdir("pos")]
    neg_files = [ngrams.ngrams(1, open("neg/"+i).read()) for i in os.listdir("neg")]
    classes = [1] * len(pos_files) + [0] * len(neg_files)
    print "Creating matrix..."
    mat = ngrams.ngrams_to_matrix(pos_files + neg_files, classes)
    print "Running classifier..."
    print validate.kfold(3, classif, mat)
    print validate.kfold(5, classif, mat)
    print validate.kfold(10, classif, mat)

if __name__ == "__main__":
    read_reviews()

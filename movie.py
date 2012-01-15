#!/usr/bin/python

import data
import ngrams
import validate
import classifier
import os
from numpy import *
from operator import itemgetter

POS_DIR="pos"
POS_POSITION_DIR="pos_position"
POS_PARTOFSPEECH_DIR="pos_tagged"
NEG_DIR="neg"
NEG_POSITION_DIR="neg_position"
NEG_PARTOFSPEECH_DIR="neg_tagged"

class MovieReviews:
    def __init__(self, clsf, n, testsize, pos_dir, neg_dir, binary=False, limit=None):
        self.classifier = clsf()
        count = 0
        pos_files = os.listdir(pos_dir)[:testsize]
        neg_files = os.listdir(neg_dir)[:testsize]
        features = {}

        self.pos_files = [{} for f in pos_files]
        self.neg_files = [{} for f in neg_files]

        # support multiple types of ngrams and limiting of ngrams
        if type(n) != type([]):
            n = [n]
        if not limit:
            limit = [0 for i in n]
        if limit and (type(limit) != type([])):
            limit = [limit]

        print "Reading files"
        for (j,lim) in zip(n,limit):
            all_grams = [ngrams.ngrams(j, open("%s/%s" % (pos_dir,i)).read()) 
                         for i in os.listdir(pos_dir)[:testsize]]
            for i in range(len(pos_files)):
                self.pos_files[i].update(all_grams[i])
            featureslist = all_grams

            all_grams = [ngrams.ngrams(j, open("%s/%s" % (neg_dir,i)).read()) 
                         for i in os.listdir(neg_dir)[:testsize]]
            for i in range(len(neg_files)):
                self.neg_files[i].update(all_grams[i])
            featureslist.extend(all_grams)

            print "Collapsing, limiting ngrams"
            features.update(ngrams.top_ngrams(ngrams.collapse_ngrams(featureslist),lim))

        print "Creating Index"
        words = set(features)
        self.classifier.addToIndex(words)
        print "# features: %s" % self.classifier.nfeatures

        print "Making classifier"        
        for i in self.pos_files:
            count += 1
            self.classifier.addFeatureVector(i, 1, binary=binary)
        for i in self.neg_files:
            self.classifier.addFeatureVector(i, -1, binary=binary)
        self.classifier.compile()
        print self.classifier.classes

classif = classifier.BayesClassifier
#classif = classifier.LinearSVMClassifier

def test(n=1,dataset='',limit=None):
    # support multiple types of ngrams
    if type(n) != type([]):
        n = [n]

    # select dataset
    if dataset=='':
        print "Using normal untagged movie dataset"
        pos_dir = POS_DIR
        neg_dir = NEG_DIR
    elif dataset=='partofspeech':
        print "Using movie dataset with part of speech tagged"
        pos_dir = POS_PARTOFSPEECH_DIR
        neg_dir = NEG_PARTOFSPEECH_DIR
    elif dataset=='position':
        print "Using movie dataset with position tagged"
        pos_dir = POS_POSITION_DIR
        neg_dir = NEG_POSITION_DIR
    elif dataset=='yelp':
        pass

    testsize=800

    print "Building Classifier"
    m = MovieReviews(classif, n, testsize, pos_dir, neg_dir, limit=limit)

    print "Testset --> Feature Vectors"
    pos_tests = None
    neg_tests = None
    for j in n:
        if pos_tests and neg_tests:
            files = os.listdir(pos_dir)[testsize:]
            for i in range(len(files)):
                pos_tests[i].update(ngrams.ngrams(j, open("%s/%s" % (pos_dir,files[i])).read()))
            files = os.listdir(neg_dir)[testsize:]
            for i in range(len(files)):
                neg_tests[i].update(ngrams.ngrams(j, open("%s/%s" % (neg_dir,files[i])).read()))
        else:
            pos_tests = [ngrams.ngrams(j, open("%s/%s" % (pos_dir,i)).read()) 
                              for i in os.listdir(pos_dir)[testsize:]]
            neg_tests = [ngrams.ngrams(j, open("%s/%s" % (neg_dir,i)).read()) 
                              for i in os.listdir(neg_dir)[testsize:]]

    print "Testing"
    pos_results = [m.classifier.classify(i) for i in pos_tests]
    pos_correct = len([i for i in pos_results if i == 1])
    print "Positive: %s of %s, %s percent" % (pos_correct,len(pos_tests),(float(pos_correct)/len(pos_tests)))
    print pos_results
    neg_results = [m.classifier.classify(i) for i in neg_tests]
    neg_correct = len([i for i in neg_results if i == -1])
    print "Negative: %s of %s, %s percent" % (neg_correct,len(neg_tests),(float(neg_correct)/len(neg_tests)))
    print neg_results
        
if __name__ == "__main__":
    test(n=[1],dataset='',limit=[2633])

# [ns]      dataset     [limits]        binary  --> +results    -results
# [2]       position    [114370]        0       --> 0.96        0.56
# [1,2]     default     [0,0]           0       --> 0.96        0.56 
# [1,2]     default     [16165,16165]   0       --> 0.94        0.71
# [1]       default     [16165]         0       --> 0.92        0.69
# [2]       default     [16165]         0       --> 0.93        0.69
# [1]       default     [2633]          0       --> 0.94        0.61

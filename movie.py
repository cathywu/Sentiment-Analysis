#!/usr/bin/python

import data
import ngrams
import validate
import classifier
import os
from numpy import *
from operator import itemgetter
from Indexes import Indexes

POS_DIR="pos"
POS_POSITION_DIR="pos_position"
POS_PARTOFSPEECH_DIR="pos_tagged"
POS_ADJ_DIR="pos_adj"

NEG_DIR="neg"
NEG_POSITION_DIR="neg_position"
NEG_PARTOFSPEECH_DIR="neg_tagged"
NEG_ADJ_DIR="neg_adj"

class MovieReviews:
    def __init__(self, clsf, n, testsize, pos_dir, neg_dir, binary=False, limit=None):

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
        self.classifier = clsf(restrictFeatures = features)

        print "# features: %s" % self.classifier.nfeatures

        print "Making classifier"        
        for i in self.pos_files:
            count += 1
            self.classifier.addFeatureVector(i, 1, binary=binary)
        for i in self.neg_files:
            self.classifier.addFeatureVector(i, -1, binary=binary)
        self.classifier.compile()

classif = classifier.BayesClassifier
#classif = classifier.LinearSVMClassifier

def test(n=1,dataset='',limit=None, binary=False):
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
    elif dataset=='adjectives':
        print "Using movie dataset with adjectives only"
        pos_dir = POS_ADJ_DIR
        neg_dir = NEG_ADJ_DIR
    elif dataset=='yelp':
        pass

    testsize=800
    iterations=1
    ind = Indexes(mode='r',iterations=iterations,train_size=testsize)

    for k in range(iterations):
        ind.next()
        print "Building Classifier"
        m = MovieReviews(classif, n, testsize, pos_dir, neg_dir, binary=binary, limit=limit)

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
        print "Positive: %s of %s, %s accuracy" % (pos_correct,len(pos_tests),(float(pos_correct)/len(pos_tests)))
        print pos_results
        neg_results = [m.classifier.classify(i) for i in neg_tests]
        neg_correct = len([i for i in neg_results if i == -1])
        print "Negative: %s of %s, %s accuracy" % (neg_correct,len(neg_tests),(float(neg_correct)/len(neg_tests)))
        print neg_results
        
if __name__ == "__main__":
    test(n=[1],dataset='adjectives',limit=[2633],binary=True)

# with testsize = 800, no shuffling
# [ns]      dataset         [limits]        binary  --> +results    -results
# [2]       position        [114370]        0       --> 0.96        0.56
# [1,2]     default         [0,0]           0       --> 0.96        0.56 
# [1,2]     default         [16165,16165]   0       --> 0.94        0.71
# [1]       default         [16165]         0       --> 0.92        0.69
# [2]       default         [16165]         0       --> 0.93        0.69
# [1]       default         [2633]          0       --> 0.94        0.61
# [1]       default         [2633]          1       --> 0.76        0.86
# [1]       default         [16165]         1       --> 0.77        0.84
# [2]       default         [16165]         1       --> 0.86        0.77
# [1,2]     default         [16165,16165]   1       --> 0.89        0.81
# [1,2]     partofspeech    [16165,16165]   1       --> 0.65        0.91
# [1]       partofspeech    [16165,16165]   1       --> 0.67        0.91
# [1]       adjectives      [2633]          1       --> 0.92        0.70
# [1]       adjectives      [2633]          0       --> 0.97        0.54
# [1]       default         [40183]         1       --> 0.83        0.79

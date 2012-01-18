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

class TestConfiguration:
    def __init__(self, clsf, n, ind, pos_dir, neg_dir, binary=False, limit=None, idf=False):
        self.count = 0
        self.n = n
        self.index = ind
        self.binary = binary
        self.limit = limit if limit else [0 for i in n]
        self.clsf = clsf
        self.idf = idf

        # filenames needed for this test configuration used
        pos_files = os.listdir(pos_dir)
        self.pos_train_data = [open("%s/%s" % (pos_dir, pos_files[i])).read() \
                                   for i in self.index.get_pos_train_ind()]
        self.pos_test_data = [open("%s/%s" % (pos_dir, pos_files[i])).read() \
                                  for i in self.index.get_pos_test_ind()]

        neg_files = os.listdir(neg_dir)
        self.neg_train_data = [open("%s/%s" % (neg_dir, neg_files[i])).read() \
                                   for i in self.index.get_neg_train_ind()]
        self.neg_test_data = [open("%s/%s" % (neg_dir, neg_files[i])).read() \
                                  for i in self.index.get_neg_test_ind()]
        self.features = {}

    def train(self):
        pos_train = [{} for f in self.pos_train_data]
        neg_train = [{} for f in self.neg_train_data]
        
        # Reading files
        for (j,lim) in zip(self.n,self.limit):
            all_grams = [ngrams.ngrams(j, f) for f in self.pos_train_data]
            for i in range(len(self.pos_train_data)):
                pos_train[i].update(all_grams[i])
            featureslist = all_grams

            all_grams = [ngrams.ngrams(j, f) for f in self.neg_train_data]
            for i in range(len(self.neg_train_data)):
                neg_train[i].update(all_grams[i])
            featureslist.extend(all_grams)

            # Collapsing, limiting ngrams
            self.features.update(ngrams.top_ngrams(ngrams.collapse_ngrams(
                        featureslist),lim))

        # Creating Index
        self.classifier = self.clsf(restrictFeatures = self.features)
        print "# features: %s" % self.classifier.nfeatures
        
        if self.idf:
            print "Using TF-IDF"
            idf = ngrams.ngrams_to_idf(pos_train + neg_train)
            for i in range(len(pos_train)):
                for j in pos_train[i]:
                    pos_train[i][j] = pos_train[i][j] * idf[j]
            for i in range(len(neg_train)):
                for j in neg_train[i]:
                    neg_train[i][j] = neg_train[i][j] * idf[j]
                            
        # Making classifier
        for i in pos_train:
            self.count += 1
            self.classifier.addFeatureVector(i, 1, binary=self.binary)
        for i in neg_train:
            self.classifier.addFeatureVector(i, -1, binary=self.binary)
        self.classifier.compile()

    def test(self):
        pos_tests = [{} for f in self.pos_test_data]
        neg_tests = [{} for f in self.neg_test_data]

        # Testset --> Feature Vectors
        for j in self.n:
            for i in range(len(self.pos_test_data)):
                pos_tests[i].update(ngrams.ngrams(j, self.pos_test_data[i]))
            for i in range(len(self.neg_test_data)):
                neg_tests[i].update(ngrams.ngrams(j, self.neg_test_data[i]))

        # Testing
        pos_results = [self.classifier.classify(i) for i in pos_tests]
        pos_correct = len([i for i in pos_results if int(i) == 1])
        print "Positive: %s of %s, %s accuracy" % (pos_correct,len(pos_tests),
                (float(pos_correct)/len(pos_tests)))
        neg_results = [self.classifier.classify(i) for i in neg_tests]
        neg_correct = len([i for i in neg_results if int(i) == -1])
        print "Negative: %s of %s, %s accuracy" % (neg_correct,len(neg_tests),
                (float(neg_correct)/len(neg_tests)))
        return (float(pos_correct)/len(pos_tests), float(neg_correct)/len(neg_tests))

class MajorityVotingTester():
    def __init__(self):
        self.classifiers = []
    def addClassifer(self):
def select_dataset(dataset):
    return {'default':(POS_DIR, NEG_DIR), #untagged
            'partofspeech':(POS_PARTOFSPEECH_DIR, NEG_PARTOFSPEECH_DIR), #part of speech tagged
            'position':(POS_POSITION_DIR, NEG_POSITION_DIR), #position tagged
            'adjectives':(POS_ADJ_DIR, NEG_ADJ_DIR) #adjectives tagged
            }[dataset]

def test(classif, n=1, train_size=500, mode='k', iterations=1, dataset='', limit=None, binary=False, idf=False):
    (pos_dir, neg_dir) = select_dataset(dataset)
    ind = Indexes(mode=mode,iterations=iterations,train_size=train_size)
    (pos_correct, neg_correct) = (0,0)
    for k in range(iterations):
        ind.next()
        m = TestConfiguration(classif, n, ind, pos_dir, neg_dir, binary=binary, limit=limit, idf=idf)
        m.train()
        (pos, neg) = m.test()
        pos_correct += pos
        neg_correct += neg
    print "Results:"
    print "Positive:", round((pos_correct/iterations)*100), "%"
    print "Negative:", round((neg_correct/iterations)*100), "%"
    print "Total:", round((neg_correct + pos_correct)/(2*iterations)*100), "%"

if __name__ == "__main__":
    #test(classifier.BayesClassifier,n=[1],train_size=800,mode='k',
    #     iterations=3,dataset='position',limit=[16165],binary=False, idf=True)
    #test(classifier.LinearSVMClassifier,n=[2],train_size=800,mode='k',
    #     iterations=3,dataset='default',limit=[16165],binary=False, idf=True)
    #test(classifier.MaximumEntropyClassifier,n=[1],train_size=800,mode='k',
    #     iterations=3,dataset='default',limit=[16165],binary=True)

    mvc = classifier.MajorityVotingClassifier()
    ind = Indexes(mode='k',iterations=3,train_size=800)
    ind.next()
    print ind
    (pos_dir, neg_dir) = select_dataset('default')
    m = TestConfiguration(classifier.BayesClassifier, [1], ind, pos_dir, neg_dir, binary=False, limit=[16165], idf=False)
    m.train()
    mvc.addClassifier(m.classifier)

    (pos_dir, neg_dir) = select_dataset('default')
    m = TestConfiguration(classifier.LinearSVMClassifier, [1], ind, pos_dir, neg_dir, binary=False, limit=[16165], idf=False)
    m.train()
    mvc.addClassifier(m.classifier)


    (pos_dir, neg_dir) = select_dataset('default')
    m = TestConfiguration(classifier.LinearSVMClassifier, [2], ind, pos_dir, neg_dir, binary=False, limit=[16165], idf=False)
    m.train()
    mvc.addClassifier(m.classifier)

    
    m.classifier = mvc
    m.test()
    exit()



# with train_size = 800, no shuffling, bayes classifier
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

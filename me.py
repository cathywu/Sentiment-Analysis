#!/usr/bin/python

import os
import ngrams
from Indexes import Indexes
import matplotlib.pyplot as plt
from classifier import MaximumEntropyClassifier

TRAIN_SIZE = 300
n = 1

print "Maximum Entropy"
pos = os.listdir("pos")
neg = os.listdir("neg")

ind = Indexes('r',1,TRAIN_SIZE)
print "> determined Indices"
ind.next()

pos_grams = [ngrams.ngrams(n, open("pos/"+pos[i]).read()) for i in ind.get_pos_train_ind()]
pos_collapsed_grams = ngrams.collapse_ngrams(pos_grams)
neg_grams = [ngrams.ngrams(n, open("neg/"+neg[i]).read()) for i in ind.get_neg_train_ind()]
neg_collapsed_grams = ngrams.collapse_ngrams(neg_grams)
print "> collapsed grams"

trainingset = [([k],'pos',v) for (k,v) in pos_collapsed_grams.iteritems()]
trainingset.extend([([k],'neg',v) for (k,v) in neg_collapsed_grams.iteritems()])
m = MaximumEntropyClassifier(trainingset)
print "> created model"

pos_res = []
neg_res = []
pos_tests = [ngrams.ngrams(n, open("pos/"+pos[i]).read()) for i in ind.get_pos_test_ind()]
pos_results = [m.classify(test) for test in pos_tests]
pos_correct = len([i for i in pos_results if i >= 0.5])
print "Positive: %s of %s, %s accuracy" % (pos_correct,len(pos_tests),(float(pos_correct)/len(pos_tests)))
print pos_results
neg_tests = [ngrams.ngrams(n, open("neg/"+neg[i]).read()) for i in ind.get_neg_test_ind()]
neg_results = [m.classify(test) for test in neg_tests]
neg_correct = len([i for i in neg_results if i < 0.5])
print "Negative: %s of %s, %s accuracy" % (neg_correct,len(neg_tests),(float(neg_correct)/len(neg_tests)))
print neg_results
print "> tested model"

# plotting results
# fig = plt.figure()
# fig.suptitle('MaxEnt results', fontsize=12)
# fig.add_subplot(1,2,1)
# plt.title('positive')
# plt.hist(pos_res)
# 
# fig.add_subplot(1,2,2)
# plt.title('negative')
# plt.hist(neg_res)
# 
# plt.show()

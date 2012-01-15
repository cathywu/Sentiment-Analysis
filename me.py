#!/usr/bin/python

from maxent import MaxentModel
import os
import ngrams
from Indexes import Indexes
import matplotlib.pyplot as plt

TRAIN_SIZE = 300
n = 1

def test_maxent():
    m = MaxentModel()
    m.verbose=1
    
    m.begin_add_event()
    m.add_event(['good'],'pos',1)
    m.add_event(['wonderful'],'pos',1)
    m.add_event(['ugly'],'neg',1)
    m.add_event(['terrible','ick'],'neg',1)
    m.end_add_event()
    
    m.train(10, "lbfgs")
    m.save("test_model",True)
    m.load("test_model")
    
    print "other label: %s" % m.eval(['mmm'],'otherlabel') # other label
    print "OOD: %s" % m.eval(['mmm'],'pos') # OOD
    print "ick: %s" % m.eval(['ick'],'pos')
    print "mmm awesome good: %s" % m.eval(['mmm','awesome','good'],'pos')
    print "mmm terrible good: %s" % m.eval(['mmm','terrible','good'],'pos')
    print "wonderful terrible good: %s" % m.eval(['wonderful','terrible','good'],'pos')

print "Maximum Entropy"
m = MaxentModel()
print "> created model"
m.verbose=1
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

m.begin_add_event()
for (k,v) in neg_collapsed_grams.iteritems():
    m.add_event([k],'neg',long(v))
for (k,v) in pos_collapsed_grams.iteritems():
    m.add_event([k],'pos',long(v))
m.end_add_event()
print "> events added"

m.train(10, 'lbfgs')
m.train(100, 'gis', 2)
print "> models trained"

pos_res = []
neg_res = []
for i in ind.get_pos_test_ind():
    pos_test_gram = ngrams.ngrams(n, open("pos/"+pos[i]).read())
    pos_res.append(m.eval(pos_test_gram.keys(),'neg'))
for i in ind.get_neg_test_ind():
    neg_test_gram = ngrams.ngrams(n, open("neg/"+neg[i]).read())
    neg_res.append(m.eval(neg_test_gram.keys(),'neg'))
print "> tested model"

# plotting results
fig = plt.figure()
fig.suptitle('MaxEnt results', fontsize=12)
fig.add_subplot(1,2,1)
plt.title('positive')
plt.hist(pos_res)

fig.add_subplot(1,2,2)
plt.title('negative')
plt.hist(neg_res)

plt.show()

#pos_collapsed_grams = ngrams.collapse_ngrams(pos_grams)
#neg_test_gram = [ngrams.ngrams(n, open("neg/"+neg[i]).read()) for i in ind.get_neg_train_ind()]
#neg_collapsed_grams = ngrams.collapse_ngrams(neg_grams)

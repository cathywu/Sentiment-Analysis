#!/usr/bin/python

import svmlight
import ngrams
import os
import pickle
import numpy
import matplotlib.pyplot as plt
from classifier import LinearSVMClassifier
import data

TRAIN_SIZE = 300
TEST_SIZE = 1000-TRAIN_SIZE
K = 3

class FeatureMap:
    """
    SVM light requires features to be identified with numbers, 
    so this object maps features (strings) to numbers
    """
    def __init__(self):
        self.fmap = {}
        self.size = 1
    def hasFeature(self,f):
        return f in self.fmap
    def getFeature(self,f):
        return self.fmap[f]
    def getID(self,id):
        return self.fmap[id]
    def addFeature(self,f):
        if f not in self.fmap:
            self.fmap[f]=self.size
            self.fmap[self.size]=f
            self.size += 1
    def getSize(self):
        return self.size

class Indexes:
    """
    Indexes object generates indices for different configurations
    Modes:
    'r' : random
    'd' : deterministic
    'k' : k-fold cross-fold validation
    """
    def __init__(self):
        self.mode = 'r'
        self.iterations = 10
    def __init__(self,mode,iterations):
        self.mode = mode
        self.iterations = iterations
        self.pos_train_ind = None 
        self.pos_test_ind = None
        self.neg_train_ind = None
        self.neg_test_ind = None
        self.gen_indices = generate_indices(mode,iterations)
    def next(self):
        (a,b,c,d) = self.gen_indices.next()
        self.pos_train_ind = a 
        self.pos_test_ind = b
        self.neg_train_ind = c
        self.neg_test_ind = d
    def get_pos_train_ind(self):
        return self.pos_train_ind
    def get_pos_test_ind(self):
        return self.pos_test_ind
    def get_neg_train_ind(self):
        return self.neg_train_ind
    def get_neg_test_ind(self):
        return self.neg_test_ind

def test_svmlight():
    training_data = [(1, [(1,2),(2,5),(3,6),(5,1),(4,2),(6,1)]),
                     (1, [(1,2),(2,1),(3,4),(5,3),(4,1),(6,1)]),
                     (1, [(1,2),(2,2),(3,4),(5,1),(4,1),(6,1)]),
                     (1, [(1,2),(2,1),(3,3),(5,1),(4,1),(6,1)]),
                     (-1, [(1,2),(2,1),(3,1),(5,3),(4,2),(6,1)]),
                     (-1, [(1,1),(2,1),(3,1),(5,3),(4,1),(6,1)]),
                     (-1, [(1,1),(2,2),(3,1),(5,3),(4,1),(6,1)]),
                     (-1, [(1,1),(2,1),(3,1),(5,1),(4,3),(6,1)]),
                     (-1, [(1,2),(2,1),(3,1),(5,2),(4,1),(6,5)]),
                     (-1, [(7,10)])]
    
    test_data = [(0, [(1,2),(2,6),(3,4),(5,1),(4,1),(6,1)]),
                 (0, [(1,2),(2,6),(3,4)])]
    
    model = svmlight.learn(training_data, type='classification', verbosity=0)
    svmlight.write_model(model, 'my_model.dat')
    predictions = svmlight.classify(model, test_data)
    for p in predictions:
        print '%.8f' % p
    # output should be 2 positive numbers

def gen_ngrams(n=2,data="pos"):
    "Generate ngrams and save locally"
    temp = []
    for i in os.listdir("%s" % data):
        temp.append(open("%s/" % data + i).read())
    temp = "\n".join(temp)
    aggregate_ngrams = ngrams.ngrams(n, temp)
    pickle.dump(aggregate_ngrams, open("%s_%sgram.dump" % (data,n),'w'))

def gen_all_ngrams():
    "Generate a bunch of ngrams for convenience"
    gen_ngrams(n=1,data="pos")
    gen_ngrams(n=1,data="neg")
    gen_ngrams(n=2,data="pos")
    gen_ngrams(n=2,data="neg")
    gen_ngrams(n=3,data="pos")
    gen_ngrams(n=3,data="neg")

def load_ngrams(n,data="pos"):
    "Load ngram data from disk"
    return pickle.load(open("%s_%sgram.dump" % (data,n)))

def gen_feature_map(strings,fmap):
    for string in strings:
        fmap.addFeature(string)

def load_features(n,fmap):
    print "Positive data"
    p = load_ngrams(n,"pos")
    v = p.values()
    upper = numpy.percentile(v,99.85)
    lower = numpy.percentile(v,65)
    print "> filtering %s values" % len(v)
    items = filter(lambda x: x[1] > lower and x[1] < upper, p.items())
    keys = [item[0] for item in items]
    print "> gen_feature_map with %s keys" % len(keys)
    gen_feature_map(keys,fmap)
    print "Negative data"
    n = load_ngrams(n,"neg")
    v = n.values()
    upper = numpy.percentile(v,99.85)
    lower = numpy.percentile(v,65)
    print "> filtering %s values" % len(v)
    items = filter(lambda x: x[1] > lower and x[1] < upper, n.items())
    keys = [item[0] for item in items]
    print "> gen_feature_map with %s keys" % len(keys)
    gen_feature_map(keys,fmap)

def training_set(ind,n=3):
    """
    
    Caution: Do not use 0 as label because it evaluates to False
    """
    pos = os.listdir("pos")
    feature_vectors = [ngrams.ngrams(n, open("pos/"+pos[i]).read()) for i in ind.get_pos_train_ind()]
    labels = [1 for i in ind.get_pos_train_ind()]
    neg = os.listdir("neg")
    feature_vectors.extend([ngrams.ngrams(n, open("neg/"+neg[i]).read()) for i in ind.get_neg_train_ind()])
    labels.extend([2 for i in ind.get_neg_train_ind()])
    (matrix, gramsdict) = ngrams.ngrams_to_matrix(feature_vectors, labels, return_gramsdict=True)
    return (matrix.asMatrix(), gramsdict)

def get_accuracy(results):
    size = len(results)/2
    pos_correct = len(numpy.nonzero(numpy.array(results[0:size]) > 0.0)[0])
    neg_correct = len(numpy.nonzero(numpy.array(results[size:]) < 0.0)[0])
    pos_accuracy = float(pos_correct)/size
    neg_accuracy = float(neg_correct)/size
    accuracy = float(pos_correct+neg_correct)/size/2
    print "Accuracy: %s (pos) %s (neg) %s (overall)" % (pos_accuracy, neg_accuracy, accuracy)
    return (pos_accuracy, neg_accuracy, accuracy)

def plot_results(results):
    size = len(results)/2
    # plot positive labels
    print "POSITIVE"
    pos_hist = numpy.histogram(p[0:size])
    print pos_hist
    fig = plt.figure()
    fig.suptitle('SVM results', fontsize=12)
    fig.add_subplot(1,2,1)
    plt.title('positive')
    plt.hist(p[0:nresults/2])
    pos_axis = plt.axis()
    
    # plot negative labels
    print "NEGATIVE"
    fig.add_subplot(1,2,2)
    plt.title('negative')
    neg_hist = numpy.histogram(p[size:])
    print neg_hist
    plt.hist(p[nresults/2:])
    neg_axis = plt.axis()
    
    # match axes of the two graphs
    low_axis = [min(a,b) for (a,b) in zip(pos_axis,neg_axis)]
    high_axis = [max(a,b) for (a,b) in zip(pos_axis,neg_axis)]
    new_axis = [low_axis[0],high_axis[1],low_axis[2],high_axis[3]]
    plt.axis(new_axis)
    plt.subplot(1,2,1)
    plt.axis(new_axis)
    
    # display plot
    plt.show()

def shuffle_ind():
    ind = numpy.arange(1000)
    from numpy.random import shuffle
    shuffle(ind)
    return ind

def generate_indices(mode='r',iterations=1):
    if mode=='d': # deterministic
        def get_indices():
            ind = numpy.arange(1000)
            pos_train_ind = ind[:TRAIN_SIZE]
            pos_test_ind = ind[TRAIN_SIZE:]
            neg_train_ind = ind[:TRAIN_SIZE]
            neg_test_ind = ind[TRAIN_SIZE:]
            for i in range(iterations):
                yield (pos_train_ind, pos_test_ind, neg_train_ind, neg_test_ind)
    elif mode=='r': # random
        def get_indices():
            for i in range(iterations):
                pos_ind = shuffle_ind()
                pos_train_ind = pos_ind[:TRAIN_SIZE]
                pos_test_ind = pos_ind[TRAIN_SIZE:]
                neg_ind = shuffle_ind()
                neg_train_ind = neg_ind[:TRAIN_SIZE]
                neg_test_ind = neg_ind[TRAIN_SIZE:]
                yield (pos_train_ind, pos_test_ind, neg_train_ind, neg_test_ind)
    elif mode=='k': # k-fold cross-validation
        pass #TODO
    return get_indices()

def run_svm(mode='r',iterations=2):
    # setup work (generate all the ngrams if they don't exist yet)
    import os
    if not os.path.isfile('pos_%sgram.dump' % n):
        gen_all_ngrams()
    
    ind = Indexes(mode,iterations)
    acc = (0,0,0)
    # run svm
    for i in range(iterations):
        ind.next()
        (train, gramsdict) = training_set(ind,n=n)
        classifier = LinearSVMClassifier(data.Data(numpy.array(train, dtype=numpy.uint16).T))
        j = ind.get_pos_test_ind()[0]
        pos = os.listdir("pos")
        test = ngrams.grams_to_featurevector(gramsdict, ngrams.ngrams(n, open("pos/"+pos[j]).read()), label=None)
        print classifier.classify(test, dtype=numpy.uint16)
        neg = os.listdir("neg")
        j = ind.get_neg_test_ind()[0]
        test = ngrams.grams_to_featurevector(gramsdict, ngrams.ngrams(n, open("neg/"+neg[j]).read()), label=None)
        print classifier.classify(test, dtype=numpy.uint16)
        print m[-2]
        print m[-1]
        # p = test_model(m,ind,n=n)
        # nresults = len(p)
        # acc = [(a+b) for (a,b) in zip(acc,get_accuracy(p))]
    print acc
    
    return (m,p)

fmap = FeatureMap()

# USAGE:
# $ ipython
# $ run -i svm
# $ get_accuracy(p)
# $ plot_results(p)

#if __name__ == "__main__":
n = 2 # specifies n in n-grams
(m,p) = (None, None)
run_svm()

# RESULTS
# 80% accuracy with TRAIN_SIZE=300 
# 84% accuracy with TRAIN_SIZE=500
# 50% accuracy with TRAIN_SIZE=900 (why?)
# Segfault with TRAIN_SIZE=100 (why?)

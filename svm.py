#!/usr/bin/python

import svmlight
import ngrams
import os
import pickle
import numpy
import matplotlib.pyplot as plt

TRAIN_SIZE = 300

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

def training_model(n=3):
    print "Loading features"
    load_features(n,fmap)
    print "Feature map size: %s" % fmap.getSize()
    print "Getting training data"
    train = []
    for i in os.listdir("pos")[0:TRAIN_SIZE]:
        train.append((1,[(fmap.getID(item[0]),item[1]) for item in ngrams.ngrams(n, open("pos/"+i).read()).items() if fmap.hasFeature(item[0])]))
    for i in os.listdir("neg")[0:TRAIN_SIZE]:
        train.append((-1,[(fmap.getID(item[0]),item[1]) for item in ngrams.ngrams(n, open("neg/"+i).read()).items() if fmap.hasFeature(item[0])]))
    print "Training model"
    model = svmlight.learn(train, type='classification', verbosity=0)
    svmlight.write_model(model, 'my_model.dat')
    return model

def test_model(model,n=3):
    test = []
    for i in os.listdir("pos")[TRAIN_SIZE:]:
        test.append((1,[(fmap.getID(item[0]),item[1]) for item in ngrams.ngrams(n, open("pos/"+i).read()).items() if fmap.hasFeature(item[0])]))
    for i in os.listdir("neg")[TRAIN_SIZE:]:
        test.append((-1,[(fmap.getID(item[0]),item[1]) for item in ngrams.ngrams(n, open("neg/"+i).read()).items() if fmap.hasFeature(item[0])]))
    predictions = svmlight.classify(model, test)
    return predictions

fmap = FeatureMap()

#if __name__ == "__main__":
m = training_model(n=1)
p = test_model(m,n=1)
nresults = len(p)

# plot positive labels
print "POSITIVE"
pos_hist = numpy.histogram(p[0:nresults/2])
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
neg_hist = numpy.histogram(p[nresults/2:])
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

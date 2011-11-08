import random
import data
from numpy import *
from PyML import *
from PyML.containers import *
from scipy.sparse import csr_matrix, lil_matrix, csc_matrix, issparse

"""
A classifier is initialized with a training set of feature vectors
and has a classify method that takes in a new vector and returns
a class
"""

class OneClassifier:
    def __init__(self, trainingset) :
        self.trainingset = trainingset
        self.classes = (True, False)
    def classify(self, point):
        return 1

class RandomClassifier:
    def __init__(self, trainingset) :
        self.trainingset = trainingset
        self.classes = (True, False)
    def classify(self, point):
        return random.randint(0,1)

class BayesClassifier:
    def __init__(self, trainingset) :
        self.classes = {}
        self.lengths = {}
        for row in trainingset.asMatrix().T:
            cls = row[-1]
            pt = row[:-1]
            if cls in self.classes:
                self.classes[cls] += pt - ones(len(pt))
            else:
                self.classes[cls] = pt
        
        for cls in self.classes:
            self.lengths[cls] = sqrt(float(dot(self.classes[cls], self.classes[cls])))
            self.classes[cls] = log(self.classes[cls])
    def classify(self, point):
        mx = 0
        mx_cls = 0
        for cls in self.classes:
            dotprod = dot(self.classes[cls], log(point)) - log(self.lengths[cls])
            if dotprod > mx:
                mx = dotprod
                mx_cls = cls
        return mx_cls
    
class BayesPresenceClassifier(BayesClassifier):
    def classify(self, point):
        return BayesClassifier.classify(self, point.clip(max=2))


class SparseBayesClassifier:
    def __init__(self, trainingset) :
        self.classes = {}
        self.lengths = {}
        self.classcounts = {}
        mat = trainingset.asSparseMatrix()
        for row in range(mat.shape[0]):
            cls = mat[row, -1]
            pt = mat[row, :-1]
            if cls in self.classes:
                self.classes[cls] += array(pt.todense()).flatten()
                self.classcounts[cls] += 1
            else:
                self.classcounts[cls] = 1
                self.classes[cls] = array(pt.todense()).flatten()
        for cls in self.classes:
            #the sparse matrix has default value 0 instead of 1, so we need to
            #stop log from breaking horrifically 
            self.classes[cls] += ones(len(self.classes[cls]))
            self.lengths[cls] = sqrt(float(dot(self.classes[cls], self.classes[cls])))
            self.classes[cls] = log(self.classes[cls])
    def classify(self, point):
        mx = 0
        mx_cls = 0
        if issparse(point):
            point = array(point.todense()).flatten()
            point += ones(len(point))
        for cls in self.classes:
            dotprod = dot(self.classes[cls], log(point)) - log(self.lengths[cls])
            if dotprod > mx:
                mx = dotprod
                mx_cls = cls
        return mx_cls

    
class LinearSVMClassifier:
    def __init__(self, trainingset):
        print "Creating dataset"
        L = [str(i) for i in trainingset.asMatrix().T[-1]]
        X = trainingset.asMatrix().T[:-1]
        data = SparseDataSet(X.T, L = L)
        print data
        self.svm = svm.SVM()
        print "Training SVM"
        self.svm.train(data)
        
    def classify(self, point):
        L= array(['1.0', '0.0'])
        X = SparseDataSet(array([point], dtype=uint16))
        print "classifying"
        return self.svm.classify(X, 0)[0]
        
        
def test_bayes():
    trainingset = data.Data(array([[2, 2, 2, 1],
                                   [1, 1, 2, 0],
                                   [1, 1, 2, 0],
                                   [2, 1, 1, 0]]).T)
    bc = BayesPresenceClassifier(trainingset)
    print bc.classify(array([2, 2, 2]))
    print bc.classify(array([3, 1, 1]))

def test_sparse_bayes():    
    trainingset_arr = array([[2, 2, 2, 1],
                              [0, 0, 2, 0],
                              [0, 1, 2, 0],
                              [1, 0, 0, 0]]).T
    trainingset = data.Data(csr_matrix(trainingset_arr))
    bc = SparseBayesClassifier(trainingset)
    print bc.classify(array([2, 2, 2]))
    print bc.classify(array([3, 1, 1]))
            
def test_svm():
    trainingset = data.Data(array([[2, 2, 2],
                                   [1, 1, 2],
                                   [1, 1, 2],
                                   [0, 1, 0]], dtype=uint16).T)
    bc = LinearSVMClassifier(trainingset)
    print bc.classify(array([2, 2, 2], dtype=uint16))
    print bc.classify(array([3, 1, 1], dtype=uint16))

if __name__ == "__main__":

    test_sparse_bayes()

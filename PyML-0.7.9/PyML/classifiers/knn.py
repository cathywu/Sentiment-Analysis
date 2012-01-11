import numpy

from PyML.classifiers.ext import knn
from PyML.evaluators import assess
from PyML.classifiers.baseClassifiers import Classifier
import time
from PyML.utils import arrayWrap

class KNN (Classifier) :
    """
    a K-Nearest-Neighbors classifier

    :Keywords:
      - `k` - the number of nearest neighbors on which to base the classification
        [default : 3]
      
    if the training data is a C++ dataset (e.g. SparseDataSet) classification
    is much faster since everything is done in C++; if a python container is
    used then it's a slower pure python implementation.
    """

    attributes = {'k' : 3}
    
    def __init__(self, arg = None, **args) :

        Classifier.__init__(self, arg, **args)

    def __repr__(self) :
        
        rep = '<' + self.__class__.__name__ + ' instance>\n'
        rep += 'number of nearest neighbors: ' + str(self.k)
            
        return rep


    def train(self, data, **args) :

        Classifier.train(self, data, **args)
        self.numClasses = data.labels.numClasses
        if data.isWrapper :
            self.knnc = knn.KNN(self.k)
            self.knnc.train(data.castToBase())
        self.data = data

        self.log.trainingTime = self.getTrainingTime()


    def classify(self, data, i) :

        '''For each class the sum of the distances to the k nearest neighbors
        is computed. The distance is computed using the given kernel'''

        # score for each class
        s = self.knnc.classScores(data.castToBase(), i)

        if self.numClasses > 2 :
            return numpy.argmax(s), max(s) - numpy.sort(s)[-2]
        elif self.numClasses == 2 :
            return numpy.argmax(s), s[0] - s[1]
        else :
            raise ValueError, 'wrong number of classes'

    def test(self, data, **args) :
	
        if data.isWrapper :
            return self.testC(data, **args)
        else :
            return assess.test(self, data, **args)

    def testC(self, data, **args) :

        testStart = time.clock()
        if data.testingFunc is not None :
            data.test(self.trainingData, **args)

        #cdecisionFunc = arrayWrap.doubleVector([])
        #cY = self.knnc.test(data.castToBase(), cdecisionFunc)
        results = self.knnc.test(data.castToBase())
        print 'lengths'
        print len(results)
        print len(data)
        Y = [int(results[i]) for i in range(len(results)/2, len(results))]
        decisionFunc = results[:len(results)/2]
        res = self.resultsObject(data, self, **args)
        for i in range(len(data)) :
            res.appendPrediction((Y[i], decisionFunc[i]), data, i)

        res.log = self.log
        try :
            computeStats = args['stats']
        except :
            computeStats = False
        if computeStats and data.labels.L is not None :
            res.computeStats()

        res.log.testingTime = time.clock() - testStart
        
        return res

    def nearestNeighbor(self, data, pattern) :

        return self.knnc.nearestNeighbor(data, pattern)
    


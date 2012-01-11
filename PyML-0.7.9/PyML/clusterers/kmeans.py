import numpy

from PyML.clusterers.ext import ckmeans
from PyML.evaluators import assess
from PyML.clusterers.baseClusterer import Clusterer
import time
from PyML.utils import arrayWrap

class Kmeans (Clusterer) :
    """
    an implementation of the kmeans clustering algorithm

    """

    def __init__(self, arg, **args) :
        
        #Clusterer.__init__(self, arg, **args)
        self.k = arg

    def __repr__(self) :
        
        rep = '<' + self.__class__.__name__ + ' instance>\n'
        rep += 'number of nearest neighbors: ' + str(self.k)
            
        return rep


    def train(self, data, **args) :

        Clusterer.train(self, data, **args)
        self.ckmeans = ckmeans.Kmeans(self.k)
        print 'calling c training'
        self.clusters = self.ckmeans.train(data.castToBase())
        #self.log.trainingTime = self.getTrainingTime()


    def classify(self, data, i) :

        pass


    def test(self, data, **args) :

        testStart = time.clock()
        if data.testingFunc is not None :
            data.test(self.trainingData, **args)

        cdecisionFunc = arrayWrap.doubleVector([])
        cY = self.ckmeans.test(data.castToBase(), cdecisionFunc)

        res.log.testingTime = time.clock() - testStart
    


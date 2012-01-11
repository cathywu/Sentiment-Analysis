import numpy
import time
import copy

from PyML.evaluators import assess,resultsObjects
from PyML.utils import misc
from PyML.base.pymlObject import PyMLobject

"""base class for for PyML classifiers"""

__docformat__ = "restructuredtext en"

containersRequiringProjection = ['VectorDataSet']

class Classifier (PyMLobject) :

    """base class for PyML classifiers, specifying the classifier api"""
    
    type = 'classifier'
    deepcopy = False

    # the type of Results object returned by testing a classifier:
    resultsObject = resultsObjects.ClassificationResults
    
    test = assess.test
    cv = assess.cv
    stratifiedCV = assess.stratifiedCV
    loo = assess.loo
    trainTest = assess.trainTest
    nCV = assess.nCV
    
    def __init__(self, arg = None, **args) :

        PyMLobject.__init__(self, arg, **args)
        if type(arg) == type('') :
            self.load(arg)
        self.log = misc.Container()

    def logger(self) :

        pass

    def __repr__(self) :

        return '<' + self.__class__.__name__ + ' instance>\n'

    def verifyData(self, data) :
        """
        verify that for a VectorDataSet the test examples refer to the same
        features used in training.
        """

        if data.__class__.__name__ != 'VectorDataSet' :
            return
        if len(misc.intersect(self.featureID, data.featureID)) != len(self.featureID) :
            raise ValueError, 'missing features in test data'

            
    def save(self, fileHandle) :

        raise NotImplementedError, 'your classifier does not implement this function'

    def train(self, data, **args) :

        # store the current cpu time:
        self._clock = time.clock()

        if not data.labels.numericLabels :
            # check if there is a class that is not represented in the training data:
            if min(data.labels.classSize) == 0 :
                raise ValueError, 'there is a class with no data'

            # store just as much about the labels as is needed:
            self.labels = misc.Container()
            self.labels.addAttributes(data.labels, ['numClasses', 'classLabels'])
        # if dealing with a VectorDataSet test data needs to have the same features
        if data.__class__.__name__ == 'VectorDataSet' :
            self.featureID = data.featureID[:]
            
        data.train(**args)
        # if there is some testing done on the data, it requires the training data:
        if data.testingFunc is not None :
            self.trainingData = data

    def trainFinalize(self) :

        self.log.trainingTime = self.getTrainingTime()

    def getTrainingTime(self) :

        return time.clock() - self._clock

    def classify(self, data, i) :

        raise NotImplementedError

    def twoClassClassify(self, data, i) :

        val = self.decisionFunc(data, i)
        if val > 0 :
            return (1, val)
        else:
            return (0, val)

class IteratorClassifier (Classifier) :

    def __iter__(self) :

        self._classifierIdx = -1
        return self

    def getClassifier(self) :

        if self._classifierIdx < 0 :
            return None
        return self.classifiers[self._classifierIdx]

    def next(self) :

        self._classifierIdx += 1
        if self._classifierIdx == len(self.classifiers) :
            raise StopIteration
        func = getattr(self.classifiers[self._classifierIdx], self._method)

        return func(self._data, **self._args)

    def test(self, data, **args) :

        self._method = 'test'
        self._data = data
        self._args = args
        return iter(self)

    def cv(self, data, **args) :

        self._method = 'cv'
        self._data = data
        self._args = args
        return iter(self)

    def stratifiedCV(self, data, **args) :

        self._method = 'stratifiedCV'
        self._data = data
        self._args = args
        return iter(self)

    def loo(self, data, **args) :

        self._method = 'loo'
        self._data = data
        self._args = args
        return iter(self)

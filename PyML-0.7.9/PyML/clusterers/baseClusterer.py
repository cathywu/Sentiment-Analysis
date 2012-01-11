import numpy
import time
import copy

from PyML.evaluators import assess
from PyML.utils import misc
from PyML.base.pymlObject import PyMLobject

"""base class for for PyML clusterers"""

__docformat__ = "restructuredtext en"

class Clusterer (PyMLobject) :

    """base class for PyML clusterers"""
    
    type = 'classifier'
    deepcopy = False
    
    def __init__(self, arg = None, **args) :

        PyMLobject.__init__(self, arg, **args)
        if type(arg) == type('') :
            self.load(arg)
        self.log = misc.Container()

    def logger(self) :

        pass

    def __repr__(self) :

        return '<' + self.__class__.__name__ + ' instance>\n'

    def save(self, fileHandle) :

        raise NotImplementedError, 'your classifier does not implement this function'

    def train(self, data, **args) :

        # store the current cpu time:
        self._clock = time.clock()
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


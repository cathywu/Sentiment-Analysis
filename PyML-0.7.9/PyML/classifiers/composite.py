
import numpy
import math

from PyML.utils import misc
from PyML.datagen import sample
from PyML.evaluators import assess
from baseClassifiers import Classifier, IteratorClassifier
import svm

__docformat__ = "restructuredtext en"

class CompositeClassifier (Classifier) :

    '''A base class for creating composite classifiers
    
    A composite classifier has an attribute called "classifier", and by default
    requests are forwarded to the appropriate function of the classifier
    (including the "test" function).
    For logging purposes, use the log attribute of the classifier rather
    than the composite log.
    See for example the FeatureSelect object.'''

    deepcopy = True

    def __init__(self, classifier, **args) :

        Classifier.__init__(self, classifier, **args)
	if type(classifier) == type('') : return
        if (not hasattr(classifier, 'type')) or classifier.type != 'classifier' :
            raise ValueError, 'argument should be a classifier'
        if classifier.__class__ == self.__class__ :
            self.classifier = classifier.classifier.__class__(
                classifier.classifier)
        else :
            self.classifier = classifier.__class__(classifier)
         
    def __repr__(self) :
        rep = '<' + self.__class__.__name__ + ' instance>\n'
        rep += 'Classifier:\n'
        rep += self.classifier.__repr__()
        
        return rep

    def preproject(self, data) :

        self.classifier.preproject(data)

    def decisionFunc(self, data, i) :

        return self.classifier.decisionFunc(data, i)

    def classify(self, data, i) :

        return self.classifier.classify(data, i)


    #def preprocess(self, data) :

    #    self.classifier.preprocess(data)

    def getTest(self) :

        return self.classifier.test

    def setTest(self) :

        raise ValueError, 'do not call this function'

    # if the classifier used by the composite implements a test function -
    # use it rather than the default assess.test
    test = property (getTest, setTest,
                     None, 'the test function of the underlying classifier')

    

class Chain (CompositeClassifier) :
    '''A chain is a list of actions to be performed on a dataset,
    the last of which is assumed to be a classifier.
    The actions can be for example a chain of preprocessing steps or
    a step of feature selection (same as using the FeatureSelect class)
    Each action in the chain is assumed to have a "train" method and is
    assumed to have a copy constructor'''

    deepcopy = True
    
    def __init__(self, arg) :
        """
        :Parameters:
          - `arg` - a Chain object of a list of objects, each of which implements
            a 'train', 'test' and has a copy constructor
        
        """
        Classifier.__init__(self)

        if arg.__class__ == self.__class__ :
            other = arg
            self.classifier = other.classifier.__class__(other.classifier)
            self.chain = [component.__class__(component)
                          for component in other.chain]
            
        elif type(arg) == type([]) :
            self.classifier = arg[-1].__class__(arg[-1])
            self.chain = [arg[i].__class__(arg[i])
                          for i in range(len(arg) - 1)]
            

    def train(self, data, **args) :

        Classifier.train(self, data, **args)
        
        for component in self.chain :
            component.train(data, **args)

        self.classifier.train(data, **args)
        self.log.trainingTime = self.getTrainingTime()
        
    def test(self, data, **args) :

        for component in self.chain :
            component.test(data, **args)

        print 'I am testing',self.classifier
        print 'testing function', self.classifier.test
        print 'the data is :', data
        return self.classifier.test(data, **args)

class FeatureSelect (CompositeClassifier) :

    """A method for combining a feature selector and classifier;
    training consists of performing feature selection and afterwards training
    the classifier on the selected features;
    use this classifier to test the accuracy of a feature selector/classifier
    combination.
    USAGE:
    construction :
    featureSelect(classifier, featureSelector)
    featureSelect(otherFeatureSelectInstance) - copy construction
    """
    
    deepcopy = True
    
    def __init__(self, arg1, arg2 = None) :

        Classifier.__init__(self)

        if arg1.__class__ == self.__class__ :
            other = arg1
            self.classifier = other.classifier.__class__(other.classifier)
            self.featureSelector = other.featureSelector.__class__(
                other.featureSelector)
        else :
            for arg in (arg1, arg2) :
                if arg.type == 'classifier' :
                    self.classifier = arg.__class__(arg)
                elif arg.type == 'featureSelector' :
                    self.featureSelector = arg.__class__(arg)
                else :
                    raise ValueError, \
                          'argument should be either classifier or featureSelector'


    def __repr__(self) :
        
        rep = '<' + self.__class__.__name__ + ' instance>\n'
        if hasattr(self, 'numFeatures') :
            rep += 'number of features trained on:' + str(self.numFeatures) + '\n'
        rep += 'Classifier:\n'
        rep += self.classifier.__repr__()
        rep += 'Feature Selector:\n'
        rep += self.featureSelector.__repr__()
            
        return rep

            
    def train(self, data, **args) :

        Classifier.train(self, data, **args)

        self.featureSelector.select(data, **args)
        #self.numFeatures = data.numFeatures
        self.classifier.log.numFeatures = data.numFeatures
        self.classifier.log.features = data.featureID[:]
        
        self.classifier.train(data, **args)
        self.classifier.log.trainingTime = self.getTrainingTime()


class FeatureSelectAll (IteratorClassifier) :

    '''A method for combining a feature selector and classifier;
    the difference from FeatureSelect is that it is specifically
    designed for computing the accuracy while varying the 
    number of features.
    '''
    
    deepcopy = True

    def __init__(self, arg1, arg2 = None) :

        Classifier.__init__(self)

        if arg1.__class__ == self.__class__ :
            other = arg1
            self.classifier = other.classifier.__class__(other.classifier)
            self.featureSelector = other.featureSelector.__class__(
                other.featureSelector)
        else :
            for arg in (arg1, arg2) :
                if arg.type == 'classifier' :
                    self.classifier = arg.__class__(arg)
                elif arg.type == 'featureSelector' :
                    self.featureSelector = arg.__class__(arg)
                else :
                    raise ValueError, \
                          'argument should be either classifier or featureSelector'


    def train(self, data, **args) :

        Classifier.train(self, data, **args)

        numFeatures = []
        n = 1
        while n < data.numFeatures :
            numFeatures.append(n)
            n *=2

        self.classifiers = [self.classifier.__class__(self.classifier)
                            for i in range(len(numFeatures))]

        featureSelector = self.featureSelector.__class__(self.featureSelector)
        rankedFeatures = featureSelector.rank(data)
	
        for i in range(len(numFeatures)) :
            selectedData = data.__class__(data)
            selectedData.keepFeatures(rankedFeatures[:numFeatures[i]])
            self.classifiers[i].train(selectedData)
            self.classifiers[i].log.numFeatures = selectedData.numFeatures

        self.classifier.log.trainingTime = self.getTrainingTime()



class AggregateClassifier (Classifier) :

    """
    classifier combines the predictions of classifiers trained on
    different datasets.
    The datasets are presented as a DataAggregate dataset container.
    """

    def __init__ (self, arg) :

        Classifier.__init__(self)
        if arg.__class__ == self.__class__ :
            self.classifiers = [classifier.__class__(classifier)
                                for classifier in arg.classifiers]
        elif type(arg) == type([]) :
            self.classifiers = [classifier.__class__(classifier)
                                for classifier in arg]

    def train(self, data, **args) :

        Classifier.train(self, data, **args)
        if not data.__class__.__name__ == 'DataAggregate' :
            raise ValueError, 'train requires a DataAggregate dataset'

        for i in range(len(self.classifiers)) :
            self.classifiers[i].train(data.datas[i], **args)
        self.log.trainingTime = self.getTrainingTime()
        
    def classify(self, data, p) :

        if not data.__class__.__name__ == 'DataAggregate' :
            raise ValueError, 'classify requires a DataAggregate dataset'

        decisionFuncs = [self.classifiers[i].decisionFunc(data.datas[i], p)
                         for i in range(len(self.classifiers))]
        #decisionFunc = numpy.sum(decisionFuncs)
        #if decisionFunc > 0 :
        #    return (1, decisionFunc)
        #else :
        #    return (0, decisionFunc)
        if decisionFuncs[0] > 0 and decisionFuncs[1] > 0 :
            return 1, numpy.sum(decisionFuncs)
        else :
            return 0, min(decisionFuncs)
        
            

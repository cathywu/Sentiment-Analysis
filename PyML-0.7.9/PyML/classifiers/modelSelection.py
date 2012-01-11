from PyML.utils import misc

from baseClassifiers import Classifier,IteratorClassifier
from composite import CompositeClassifier
from PyML.containers import ker
from PyML.classifiers import svm

'''classes for model selection'''

__docformat__ = "restructuredtext en"


class Param (IteratorClassifier) :

    """
    A class for training a classifier with several values of a parameter.
    Training trains a classifier for each value of the parameter.
    Testing returns a list evaluating each trained classifier on the given
    dataset.
    
    Example::
    
      p = Param(svm.SVM(), 'C', [0.1, 1, 10, 100, 1000])
    """

    def __init__(self, arg, attribute = 'C', values = [0.1, 1, 10, 100, 1000]) :
        """
        :Parameters:
          - `arg` - another Param object, or the classifier to be used
          - `attribute` - the attribute of the classifier that needs tuning
          - `values` - a list of values to try
        """

        if arg.__class__ == self.__class__ :
            other = arg
            self.attribute = other.attribute
            self.values = other.values[:]
            self.classifiers = [classifier.__class__(classifier)
                                for classifier in other.classifiers]
            for i in range(len(self)) :
                misc.mysetattr(self.classifiers[i], self.attribute, self.values[i])
        elif hasattr(arg, 'type') and arg.type == 'classifier' :
            self.attribute = attribute
            self.values = values
            self.classifiers = [arg.__class__(arg)
                                for i in range(len(self.values))]
            for i in range(len(self)) :
                misc.mysetattr(self.classifiers[i], self.attribute, self.values[i])
        elif type(arg) == type([]) :
            self.classifiers = [arg[i].__class__(arg[i])
                                for i in range(len(arg))]

    def __len__(self) :

        return len(self.classifiers)

    def __repr__(self) :
        
        rep = '<' + self.__class__.__name__ + ' instance>\n'
        rep += 'classifier:\n'
        rep += self.classifiers[0].__repr__()
        rep += 'attribute: %s\n' % self.attribute
        rep += 'values:' + str(self.values) + '\n'

        return rep


    def train(self, data, **args) :

        for classifier in self.classifiers :
            classifier.train(data, **args)
        #self.log.trainingTime = self.getTrainingTime()
        
        
class ParamGrid (Param) :
    """
    A class for training and testing a classifier on a grid of parameter
    values for two attributes of the classifier.
    
    Example::

      p = ParamGrid(svm.SVM(ker.Gaussian()), 'C', [0.1, 1, 10, 100, 1000], 
                    'kernel.gamma', [0.001, 0.01, 0.1, 1, 10])
    """

    def __init__(self, arg,
                 attribute1 = 'C', values1 = [0.1, 1, 10, 100, 1000],
                 attribute2 = 'kernel.gamma', values2 = [0.001, 0.01, 0.1, 1, 10]) :

        """
        :Parameters:
          - `arg` - another Param object, or the classifier to be used
          - `attribute1` - the first attribute of the classifier that needs tuning
          - `values1` - a list of values to try for attribute1
          - `attribute2` - the second attribute 
          - `values2` - a list of values to try for attribute2
          
        """


        if arg.__class__ == self.__class__ :
            other = arg
            self.attribute1 = other.attribute1
            self.values1 = other.values1[:]
            self.attribute2 = other.attribute2
            self.values2 = other.values2[:]
            self.classifiers = [classifier.__class__(classifier)
                                for classifier in other.classifiers]
        elif hasattr(arg, 'type') and arg.type == 'classifier' :
            self.attribute1 = attribute1
            self.values1 = values1
            self.attribute2 = attribute2
            self.values2 = values2
            
            self.classifiers = [arg.__class__(arg)
                                for i in range(len(values1) * len(values2))]

        for i in range(len(self.values1)) :
            for j in range(len(self.values2)) :
                classifierID = i * len(self.values2) + j
                misc.mysetattr(self.classifiers[classifierID],
                               self.attribute1,
                               self.values1[i])
                misc.mysetattr(self.classifiers[classifierID],
                               self.attribute2,
                               self.values2[j])


    def __repr__(self) :
        
        rep = '<' + self.__class__.__name__ + ' instance>\n'
        rep += 'classifier:\n'
        rep += self.classifiers[0].__repr__()
        rep += 'attribute1: %s\n' % self.attribute1
        rep += 'values1:' + str(self.values1) + '\n'
        rep += 'attribute2: %s\n' % self.attribute2
        rep += 'values2:' + str(self.values2) + '\n'

        return rep


class ModelSelector (CompositeClassifier) :
    """
    A model selector decides on the best classifier parameters
    using the param object it receives as input.
    Parameters are chosen according to the success rate in CV (or success
    on a dataset provided to the train method.

    """

    attributes = {'numFolds' : 5,
                  'measure' : 'balancedSuccessRate',
                  'foldsToPerform' : 5,}

    def __init__(self, arg, **args) :
        """
        :Parameters:
          - `arg` - another ModelSelector or a Param object

        :Keywords:
          - `measure` - which measure of accuracy to use for selecting the
            best classifier (default = 'balancedSuccessRate')
            supported measures are: 'balancedSuccessRate', 'successRate',
            'roc', 'roc50' (you can substitute any number instead of 50)
          - `numFolds` - number of CV folds to use when performing model selection
          - `foldsToPerform` - the number of folds to actually perform
        """
        
        
        Classifier.__init__(self, **args)

        if arg.__class__ == self.__class__ :
            self.param = arg.param.__class__(arg.param)
            self.measure = arg.measure
            self.numFolds = arg.numFolds
        elif arg.__class__.__name__.find('Param') >= 0 :
            self.param = arg.__class__(arg)
        else :
            raise ValueError, 'wrong type of input for ModelSelector'
        
        self.classifier = None

    def __repr__(self) :
        
        rep = '<' + self.__class__.__name__ + ' instance>\n'
        if self.classifier is not None :
            rep += self.classifier.__repr__()
        else :
            rep += self.param.__repr__()

        return rep


    def train(self, data, **args) :
        """
        :Keywords:
          - `train` - boolean - whether to train the best classifier
            (default: True)
        """

        Classifier.train(self, data, **args)

        maxSuccessRate = 0
        bestClassifier = None
        classifierIdx = 0
        args['numFolds'] = self.numFolds
        args['foldsToPerform'] = self.foldsToPerform
        
        for r in self.param.stratifiedCV(data, **args) :
            successRate = getattr(r, self.measure)
            if successRate > maxSuccessRate :
                bestClassifier = classifierIdx
                maxSuccessRate = successRate
            classifierIdx += 1

        self.log.maxSuccessRate = maxSuccessRate
        
        self.classifier = self.param.classifiers[bestClassifier].__class__(
            self.param.classifiers[bestClassifier])

        if 'train' not in args or args['train'] is True :
            self.classifier.train(data, **args)
        
        self.classifier.log.trainingTime = self.getTrainingTime()
        self.classifier.log.classifier = self.classifier.__class__(self.classifier)


    def save(self, fileHandle) :

        self.classifier.save(fileHandle)


class SVMselect (ModelSelector) :
    """
    A model selector for searching for best parameters for an
    SVM classifier with a Gaussian kernel
    Its search strategy is as follows:
    First optimize the width of the Gaussian (gamma) for a fixed (low)
    value of C, and then optimize C.
    """
    
    attributes = {'C' : [0.01, 0.1, 1, 10, 100, 1000],
                  'gamma' : [0.001, 0.01, 0.1, 1, 10],
                  'Clow' : 10,
                  'numFolds' : 5,
                  'measure' : 'balancedSuccessRate'}

    def __init__(self, arg = None, **args) :
        """
        :Parameters:
          - `arg` - another ModelSelector object

        :Keywords:
          - `C` - a list of values to try for C
          - `gamma` - a list of value to try for gamma
          - `measure` - which measure of accuracy to use for selecting the
            best classifier (default = 'balancedSuccessRate')
            supported measures are: 'balancedSuccessRate', 'successRate',
            'roc', 'roc50' (you can substitute another number instead of 50)
          - `numFolds` - number of CV folds to use when performing model selection
        """

        Classifier.__init__(self, arg, **args)

        self.classifier = None

    def __repr__(self) :
        
        rep = '<' + self.__class__.__name__ + ' instance>\n'
        if self.classifier is not None :
            rep += self.classifier.__repr__()
        rep += 'C: ' + str(self.C) + '\n'
        rep += 'gamma: ' + str(self.gamma) + '\n'

        return rep

    def train(self, data, **args) :
        """
        :Keywords:
          - `train` - boolean - whether to train the best classifier
            (default: True)
          - `vdata` - data to use for testing instead of using cross-validation
            (not implemented yet)
        """
        Classifier.train(self, data, **args)

        kernel = ker.Gaussian()
        gammaSelect = ModelSelector(Param(svm.SVM(kernel, C = self.Clow),
                                          'kernel.gamma', self.gamma),
                                    measure = self.measure,
                                    numFolds = self.numFolds)
        gammaSelect.train(data)

        kernel = ker.Gaussian(gamma = gammaSelect.classifier.kernel.gamma)
        cSelect = ModelSelector(Param(svm.SVM(kernel), 'C', self.C),
                                measure = self.measure,
                                numFolds = self.numFolds)
        cSelect.train(data)
        
        self.classifier = cSelect.classifier.__class__(cSelect.classifier)

        if 'train' not in args or args['train'] is True :
            self.classifier.train(data, **args)
        
        self.classifier.log.trainingTime = self.getTrainingTime()
        self.classifier.log.classifier = self.classifier.__class__(self.classifier)
        


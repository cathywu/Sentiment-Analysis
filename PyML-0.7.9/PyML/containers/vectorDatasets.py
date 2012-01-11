
import numpy

from PyML.containers.baseDatasets import WrapperDataSet, BaseVectorDataSet
from PyML.utils import arrayWrap,misc
from ext import csparsedataset,cvectordataset

class BaseCVectorDataSet (WrapperDataSet, BaseVectorDataSet) :
    """A base class for vector dataset containers implemented in C++"""

    def __init__(self) :
        if self.__class__.__name__ == 'SparseDataSet' :		
            self.container = csparsedataset.SparseDataSet
        elif self.__class__.__name__ == 'VectorDataSet' :
            self.container = cvectordataset.VectorDataSet

    def copy(self, other, patterns, deepcopy) :
        """
        copy a wrapper dataset

        :Parameters:
          - `other` - the other dataset
          - `patternsToCopy` - a list of patterns to copy
          - `deepcopy` - a 0/1 flag telling whether to do deepcopy or not
        """
    
        if patterns is None :
            patterns = range(len(other))
        self.container.__init__(self, other, patterns)
        self.featureDict = other.featureDict.copy()
        self.featureID = other.featureID[:]

        
    def initializeDataMatrix(self, numPatterns, numFeatures) :

        self.container.__init__(self, numPatterns)


    def addPattern(self, x, i) :

        if type(x) == type({}) :
            keys,values = arrayWrap.dict2vectors(x)
        elif type(x) == type(numpy.array(1)) or type(x) == type([]) :
            keys = arrayWrap.longVector([])
            values = arrayWrap.doubleVector(x)
        else:
            raise TypeError,"data vectors must be dictionary, list or arrays"
        self.container.addPattern(self, keys, values)

    def addFeature(self, id, values) :
        """
        Add a feature to a dataset.

        :Parameters:
          - `id` - the id of the feature
          - `values` - list of values

        """
        if len(values) != self.size() :
            raise ValueError, \
                'number of values provided does not match dataset size'
        if type(id) == type(1) : 
            id = str(id)
        hashID = hash(id)
        if not hasattr(self, 'featureKeyDict') :
            self.addFeatureKeyDict()
        if hashID in self.featureKeyDict :
            raise ValueError, 'Feature already exists, or hash clash'
        if type(values) != type([]) :
            values = [v for v in values]

        self.container.addFeature(self, hashID, values)
        self.updateFeatureDict(id)
        
    def addFeatures(self, other) :
        """
        Add features to a dataset using the features in another dataset

        :Parameters:
          - `other` - the other dataset
        """

        if len(other) != len(self) :
            raise ValueError, 'number of examples does not match'
        if not hasattr(self, 'featureKeyDict') :
            self.addFeatureKeyDict()
        for id in other.featureID :
            if hash(id) in self.featureKeyDict :
                raise ValueError, 'Feature already exists, or hash clash'
        self.container.addFeatures(self, other)
        self.updateFeatureDict(other)


    def getPattern(self, i) :

        if i < 0 or i >= len(self) :
            raise ValueError, 'Index out of range'
        return self.container.getPattern(self, i)
        
    def extendX(self, other, patterns) :

        self.container.extend(self, other, patterns)

    def eliminateFeatures(self, featureList):
        """eliminate a list of features from a dataset
        INPUT:
        featureList - a list of features to eliminate; these are numbers
        between 0 and numFeatures-1 (indices of features, not their IDs)"""

        if len(featureList) == 0 : return
        if type(featureList[0]) == type('') :
            featureList = self.featureNames2IDs(featureList)
        featureList.sort()
        if type(featureList) != list :
            featureList = list(featureList)
        if max(featureList) >= self.numFeatures or min(featureList) < 0 :
            raise ValueError, 'Bad feature list'
        cfeatureList = arrayWrap.intVector(featureList)
        self.container.eliminateFeatures(self, cfeatureList)
        self.updateFeatureDict(featureList)
        
    def scale(self, w) :
        """rescale the columns of the data matrix by a weight vector w:
        set X[i][j] = X[i][j] * w[j]
        """

        if type(w) == type(1.0) :
            w = [w for i in range(self.numFeatures)]
        if type(w) != type([]) :
            w = list(w)
            #numpy.ones(self.numFeatures, numpy.float_) * w
        self.container.scale(self, w)

    def translate(self, c) :
        
        if type(c) != type([]) :
            c = list(c)
        self.container.translate(self, c)

    def mean(self, patterns = None) :

        if patterns is None : patterns = range(len(self))
        if min(patterns) < 0 or max(patterns) >= len(self) :
            raise ValueError, 'Pattern index out of range'
        cpatterns = arrayWrap.intVector(patterns)
        return self.container.mean(self, cpatterns)

    def std(self, patterns = None) :

        if patterns is None : patterns = range(len(self))
        if type(patterns) != type([]) : patterns = list(patterns)
        if min(patterns) < 0 or max(patterns) >= len(self) :
            raise ValueError, 'Pattern index out of range'
        cpatterns = arrayWrap.intVector(patterns)
        return self.container.standardDeviation(self, cpatterns)

    def featureCount(self, feature, patterns = None) :

        if patterns is None : patterns = range(len(self))
        if type(patterns) != type([]) : patterns = list(patterns)    
        if min(patterns) < 0 or max(patterns) >= len(self) :
            raise ValueError, 'Pattern index out of range'
        cpatterns = arrayWrap.intVector(patterns)
        return self.container.featureCount(self, feature, cpatterns)

    def featureCounts(self, patterns = None) :

        if patterns is None : patterns = range(len(self))
        if type(patterns) != type([]) : patterns = list(patterns)        
        if min(patterns) < 0 or max(patterns) >= len(self) :
            raise ValueError, 'Pattern index out of range'
        cpatterns = arrayWrap.intVector(patterns)
        return self.container.featureCounts(self, cpatterns)

    def nonzero(self, feature, patterns = None) :

        if patterns is None : patterns = range(len(self))
        if type(patterns) != type([]) : patterns = list(patterns)        
        if min(patterns) < 0 or max(patterns) >= len(self) :
            raise ValueError, 'Pattern index goes outside of range'
        cpatterns = arrayWrap.intVector(patterns)
        return self.container.nonzero(self, feature, cpatterns)

    def commonFeatures(self, pattern1, pattern2) :

        return [self.featureKeyDict[featureKey] for featureKey in
                self.container.commonFeatures(self, pattern1, pattern2)]
        
    def normalize(self, norm=2) :

        norm = int(norm)
        if norm not in [1,2] :
            raise ValueError, 'bad value for norm'
        self.container.normalize(self, norm)


class VectorDataSet (BaseCVectorDataSet, cvectordataset.VectorDataSet) :

    def __init__(self, arg = None, **args):
        BaseCVectorDataSet.__init__(self)
        BaseVectorDataSet.__init__(self, arg, **args)

    def addPattern(self, x, i) :

        if type(x) == type(numpy.array(1)) or type(x) == type([]) :
            values = arrayWrap.doubleVector(x)
        else:
            raise TypeError, "data vectors must be list or array"
        self.container.addPattern(self, values)
        

    def updateFeatureDict(self, arg = None) :

        if arg.__class__ == self.__class__ :   
            # features were extended with those in another dataset
            other = arg
            self.featureID.extend(other.featureID)
        elif type(arg) == list :
            print 'recalculating feature ID'
            #features were eliminated:
            eliminated = misc.list2dict(arg)
            self.featureID = [self.featureID[i] for i in range(len(self.featureID))
                              if i not in eliminated]
        elif type(arg) == type(1) or type(arg) == type('') :
            # a feature was added
            id = arg
            self.featureID.append(id)
            self.featureDict[id] = self.numFeatures - 1
            return

        self.featureDict = {}
        for i in range(len(self.featureID)) :
            self.featureDict[self.featureID[i]] = i

class SparseDataSet (BaseCVectorDataSet, csparsedataset.SparseDataSet) :

    def __init__(self, arg = None, **args):
        BaseCVectorDataSet.__init__(self)
        BaseVectorDataSet.__init__(self, arg, **args)

    def updateFeatureDict(self, arg = None) :
        
        if arg.__class__ == self.__class__ :
            other = arg
            self.featureID.extend(other.featureID)
            self.featureID.sort(cmp = lambda x,y : cmp(hash(x), hash(y)))
        elif type(arg) == list :
            #features were eliminated:
            eliminated = misc.list2dict(arg)
            self.featureID = [self.featureID[i] for i in range(len(self.featureID))
                              if i not in eliminated]
        elif type(arg) == type(1) or type(arg) == type('') :
            # a feature was added:
            id = arg
            self.featureID.append(id)
            self.featureID.sort(cmp = lambda x,y : cmp(hash(x), hash(y)))

        self.featureDict = {}
        self.featureKeyDict = {}
        for i in range(len(self.featureID)) :
            self.featureDict[self.featureID[i]] = i
            self.featureKeyDict[hash(self.featureID[i])] = i

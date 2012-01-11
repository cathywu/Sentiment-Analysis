
from PyML.utils import arrayWrap,misc,myio
from PyML.containers import ker
from ext import ckernel
from ext import caggregate
from ext import csparsedataset
from ext import cvectordataset
from ext import ckerneldata
from ext import csequencedata

import string
import numpy
import math
import copy
import random

import parsers

"""data container classes that are pure python (not in use at the moment)"""

__docformat__ = "restructuredtext en"


class PySparseDataSet (BaseVectorDataSet):
    """A sparse dataset container"""

    def __len__(self) :

        return len(self.X)
    
    def getNumFeatures(self) :

        return len(self.featureID)

    def setNumFeatures(self, value) :

        raise ValueError, 'do not call this function!'

    numFeatures = property (getNumFeatures, setNumFeatures,
                            None, 'The number of features in a dataset')
    
    def copy(self, other, patternsToCopy, deepcopy) :
        """copy the X variable of a sparse dataset
        INPUT:
        other - the other dataset
        patternsToCopy - a list of patterns to copy
        deepcopy - a 0/1 flag telling whether to do deepcopy or not"""
    
        X = None

        if patternsToCopy is None :
            patternsToCopy = range(len(other))

        featureKeyDict = {}
        if other.X is not None :
            X = []
            for i in patternsToCopy:
                if deepcopy :
                    X.append(copy.deepcopy(other.X[i]))
                else :
                    X.append(other.X[i])
                if len(patternsToCopy) < len(other) :
                    for featureKey in other.X[i] :
                        featureKeyDict[featureKey] = 1
                        
        if len(patternsToCopy) == len(other) :
            self.featureKeyDict = copy.deepcopy(other.featureKeyDict)
            self.featureKey = other.featureKey[:]
            self.featureID = other.featureID[:]
        else :
            self.featureKey = featureKeyDict.keys()
            self.featureKey.sort()
            self.featureKeyDict = {}
            for i in range(len(self.featureKey)) :
                self.featureKeyDict[self.featureKey[i]] = i
            self.featureID = [other.featureID[i] for i in range(other.numFeatures) 
                              if other.featureKey[i] in self.featureKeyDict]

        self.X = X
        #self._numFeatures = len(self.featureID)


    def initializeDataMatrix(self, numPatterns, numFeatures) :

        self.X = []

    def addPattern(self, x, i) :

        if type(x) == type({}) :
            self.X.append(x)
        else :
            xDict = {}
            for i in range(len(x)) :
                xDict[i] = x[i]
            self.X.append(xDict)

    def addFeature(self, id, values) :

        hashID = hash(id)
        if hashID in self.featureKeyDict :
            raise ValueError, 'feature already exists, or hash problem'
        for i in range(len(self)) :
            if values[i] != 0 :
                self.X[i][hashID] = values[i]
            
        # update the featureKey, featureID attributes:
        pos = numpy.searchsorted(self.featureKey, hashID)
        self.featureKey.insert(pos, hashID)
        self.featureID.insert(pos, id)
        self.featureKeyDict = misc.list2dict(self.featureKey, range(len(self.featureKey)))


    def getPattern(self, i) :

        return self.X[i]

    def featureIDcompute(self) :

        pass

    def extendX(self, other, patterns) :
        
        for p in patterns :
            self.X.append(other.X[p])


    def eliminateFeatures(self, featureList):
        """eliminate a list of features from a dataset
        INPUT:
        featureList - a list of features to eliminate; these are numbers
        between 0 and numFeatures-1 (indices of features, not their IDs)"""

        if len(featureList) == 0 : return

        if self.verbose :
            print 'eliminating features...'

        if type(featureList[0]) == type('') :
            featureList = self.featureNames2IDs(features)
            
        elimDict = {}
        for feature in featureList :
            elimDict[self.featureKey[feature]] = 1

        featureKeyDict = {}
        for i in range(len(self)) :
            if self.verbose and i % 1000 == 0 and i > 0 :
                print i
            featureKeys = self.X[i].keys()
            for featureKey in featureKeys :
                if featureKey in elimDict :
                    del self.X[i][featureKey]
                else :
                    featureKeyDict[featureKey] = 1

        oldFeatureKey = self.featureKey
        self.featureKey = featureKeyDict.keys()
        self.featureKey.sort()
        self.featureKeyDict = {}
        for i in range(len(self.featureKey)) :
            self.featureKeyDict[self.featureKey[i]] = i
        self.featureID = [self.featureID[i] for i in range(len(self.featureID))
                          if oldFeatureKey[i] in self.featureKeyDict]
                    

    def featureView(self) :
        """F is a list where F[i] is a dictionary whose entries are the non
        zero entries of feature number i:
        F[self.featureKeyDict[f]][i] = X[i][f]        
        """
        
        F = [{} for i in range(self.numFeatures)]

        for i in range(len(self)) :
            for f in self.X[i].keys() :
                F[self.featureKeyDict[f]][i] = self.X[i][f]

        self.F = F


    def getFeature(self, feature, patterns = None) :

        if patterns is None :
            patterns = range(len(self))
        values = numpy.zeros(len(patterns), numpy.float_)
        for i in range(len(patterns)) :
            if self.featureKey[feature] in self.X[patterns[i]] :
                values[i] = self.X[patterns[i]][self.featureKey[feature]]

        return values
                

    def dotProduct(self, x, y, other = None) : 

        if type(x) == type(1) :
            x = self.X[x]
            if other is not None :
                y = other.X[y]
            else :
                y = self.X[y]
        sum = 0.0
        xKeys = x.keys()
        for xKey in xKeys :
            if y.has_key(xKey) :
                sum += y[xKey] * x[xKey]
        return sum

    def norm(self, pattern, p = 1) :

        sum = 0.0
        for xKey in self.X[pattern] :
            if p == 1 :
                sum += abs(self.X[pattern][xKey])
            elif p == 2 :
                sum += self.X[pattern][xKey] * self.X[pattern][xKey]
            else :
                raise ValueError, 'wrong value for p'

        if p == 1 :
            return sum
        else :
            return math.sqrt(sum)
        
    def normalize(self, p = 1) :
        """normalize dataset according to the p-norm, p=1,2"""

        for i in range(len(self)) :
            norm = self.norm(i, p)
            if norm == 0 : continue
            for xKey in self.X[i] :
                self.X[i][xKey] /= norm
        
            
    def scale(self, w) :
        """rescale the columns of the data matrix by a weight vector w:
        set X[i][j] = X[i][j] * w[j]
        w is either a dictionary or an array
        """

        if type(w) != type({}) :
            wDict = {}
            for i in range(self.numFeatures) :
                wDict[self.featureKey[i]] = w[i]
            w = wDict
        for i in range(len(self)) :
            for featureKey in self.X[i] :
                if featureKey in w :
                    self.X[i][featureKey] *= w[featureKey]
                else :
                    self.X[i][featureKey] = 0.0
                  

    def mean(self, patterns = None) :

        if patterns is None : patterns = range(len(self))
        
        featureMean = numpy.zeros(self.numFeatures, numpy.float_)

        for i in patterns :
            for featureKey in self.X[i] :
                featureMean[self.featureKeyDict[featureKey]] += self.X[i][featureKey]

        return featureMean / len(patterns)
        
    def translate(self, translation) :
        """subtract the input array from the data.
        the sparsity of the data is not altered, ie, zero entries are not
        made nonzero by the translation
        """
        for i in range(len(self)) :
            for featureKey in self.X[i] :
                self.X[i][featureKey] -= translation[self.featureKeyDict[featureKey]]
        
    def std(self, patterns = None) :
        
        if patterns is None : patterns = range(len(self))
        
        featureSq = numpy.zeros(self.numFeatures, numpy.float_)

        for i in patterns :
            for featureKey in self.X[i] :
                featureSq[self.featureKeyDict[featureKey]] += self.X[i][featureKey]**2

        featureVar = featureSq / float(len(patterns)) - self.mean(patterns)**2
            
        return numpy.sqrt(numpy.clip(featureVar, 0, 1e10))

    def featureCount(self, feature, patterns = None) :
        
        if patterns is None :
            patterns = range(len(self))

        count = 0
        featureKey = self.featureKey[feature]
        for i in patterns :
            if data.X[i].has_key(featureKey) and data.X[i][featureKey] != 0 :
                count += 1
        
        return count
        
    def featureCounts(self, patterns = None) :
        
        if patterns is None :
            patterns = range(len(self))

        counts = numpy.zeros(self.numFeatures, numpy.float_)
        for i in patterns :
            for featureKey in data.X[i] :
                feature = data.featureKeyDict[featureKey]
                if data.X[i][featureKey] != 0 :
                    counts[feature] += 1
        
        return counts




class PyVectorDataSet (BaseVectorDataSet) :
    """A non-sparse dataset container; uses a numpy array"""

    def __len__(self) :
        """the number of patterns in the dataset"""

        if self.X is not None :
            return len(self.X)
        else :
            raise ValueError, "no data here!"

    def getNumFeatures(self) :

        return len(self.featureID)

    def setNumFeatures(self, value) :

        raise ValueError, 'do not call this function!'

    numFeatures = property (getNumFeatures, setNumFeatures,
                            None, 'The number of features in a dataset')

    def fromArrayAdd(self, X) :

        self.X = X

    def dotProduct(self, x, y, other = None) :

        if type(x) == type(1) :
            x = self.X[x]
            if other is not None :
                y = other.X[y]
            else :
                y = self.X[y]

        return numpy.dot(x, y)
        
    def initializeDataMatrix(self, numPatterns, numFeatures) :

        self.X = numpy.zeros((numPatterns, numFeatures), numpy.float_)

    def addPattern(self, x, i) :

        for j in range(len(x)) :
            self.X[i][j] = x[j]

    def getPattern(self, i) :

        return self.X[i]
    
    def extendX(self, other, patterns) :

        X = self.X
        self.X = numpy.zeros((len(self) + len(patterns), len(self.numFeatures)),
                               numpy.float_)
        for i in range(len(X)) :
            self.X[i] = X[i]
        for i in patterns :
            self.X[i + len(X)] = other.X[i]

    def featureIDcompute(self) :

        pass

    def copy(self, other, patternsToCopy, deepcopy) :
        """deepcopy is performed by default, so the deepcopy flag is ignored"""

        X = None
        K = None
        numFeatures = None
        if patternsToCopy is None :
            patternsToCopy = range(len(other))
        else :
            # keep track of the original IDs of the patterns:
            if hasattr(other, 'origID') :
                self.origID = [other.origID[p] for p in patternsToCopy]
            else :
                self.origID = patternsToCopy[:]
            
        if other.X is not None :
            numFeatures = other.numFeatures
            X = numpy.take(other.X, patternsToCopy)

        self.X = X
        self.featureID = other.featureID[:]
        self.featureKey = other.featureKey[:]
        self.featureKeyDict = copy.deepcopy(other.featureKeyDict)
        
        #self._numFeatures = numFeatures

    def eliminateFeatures(self, featureList) :
        """eliminate a list of features from a dataset
        Input:
        featureList - a list of features to eliminate; these are numbers
        between 0 and numFeatures-1 (indices of features, not their IDs)"""

        if len(featureList) == 0 : return
        if type(featureList[0]) == type('') :
            featureList = self.featureNames2IDs(features)
        featuresToTake = misc.setminus(range(self.numFeatures), featureList)
        featuresToTake.sort()
        self.featureID = [self.featureID[i] for i in featuresToTake]
        self.featureKey = [self.featureKey[i] for i in featuresToTake]
        self.featureKeyDict = {}
        for i in range(len(self.featureKey)) :
            self.featureKeyDict[self.featureKey[i]] = i        
        
        self.X = numpy.take(self.X, featuresToTake, 1)
        #self._numFeatures -= len(featureList)


    def getFeature(self, feature, patterns = None) :

        if patterns is None :
            patterns = range(len(self))
        values = numpy.zeros(len(patterns), numpy.float_)
        for i in range(len(patterns)) :
            values[i] = self.X[i][feature]

        return values

    def norm(self, pattern, p = 1) :

        if p == 1 :
            return numpy.sum(numpy.absolute(self.X[pattern]))
        elif p == 2 :
            return math.sqrt(numpy.sum(numpy.dot(self.X[pattern])))
        else :
            raise ValueError, 'wrong value of p'

    def normalize(self, p = 1) :
        """normalize dataset according to the p-norm, p=1,2"""
        
        for i in range(len(self)) :
            norm = self.norm(i, p)
            if norm == 0 : continue
            self.X[i] = self.X[i] / norm

    def scale(self, w) :
        """rescale the columns of the data matrix by a weight vector w:
        set X[i][j] = X[i][j] / w[j]
        """
        
        self.X = self.X * w

    def translate(self, c) :

        self.X = self.X - numpy.resize(c, (len(self), len(c)))

    def mean(self, patterns = None) :

        if patterns is None or len(patterns) == len(self) :
            return numpy.mean(self.X)

        featureMean = numpy.zeros(self.numFeatures, numpy.float_)

        for i in patterns :
            featureMean += self.X[i]

        return featureMean / len(patterns)

        
    def std(self, patterns = None) :
        
        if patterns is None or len(patterns) == len(self) :
            return numpy.std(self.X) * len(self) / (len(self) - 1)
        
        featureSq = numpy.zeros(self.numFeatures, numpy.float_)

        for i in patterns :
            featureSq += self.X[i]**2

        featureVar = featureSq / float(len(patterns)) - self.mean(patterns)**2
            
        return numpy.sqrt(numpy.clip(featureVar, 0, 1e10))

    def featureCount(self, feature, patterns = None) :

        if patterns is None :
            patterns = range(len(self))

        count = 0
        for p in patterns :
            if data.X[p][feature] != 0 : count+=1
        
        return count

    def featureCounts(self, patterns = None) :

        if patterns is None :
            patterns = range(len(self))
        
        counts = numpy.zeros(self.numFeatures)
        for i in patterns :
            counts += numpy.not_equal(data.X[i], 0)

        return counts

    def csvwrite(self, fileName, delim = ' ', idCol = -1) :

        fileHandle = open(fileName, 'w')
        if self.labels.numClasses == 2 :
            Y = [self.labels.Y[i] * 2 - 1 for i in range(len(self))]
        else :
            Y = self.labels.Y
            
        for i in range(len(self)) :
            outstr = ''
            for j in range(self.numFeatures) :
                outstr += str(self.X[i][j]) + delim
            fileHandle.write(outstr + str(Y[i]) + '\n')
        fileHandle.close()
        



from PyML.base.pymlObject import PyMLobject
from PyML.utils import misc
from PyML.feature_selection import featsel
import numpy
import math

import random

def pca(X, numcomp = None) :
    '''returns the matrix X as represented in the numcomp leading principal
    components
    if numcomp is None, all principal components are returned'''
    
    d = numpy.shape(X)[1]
    if numcomp is None :
        numcomp = d
        
    [u,s,v] = numpy.linalg.svd(X)
    v = numpy.transpose(v)
    v = v[:,:numcomp]
    print numpy.shape(X)
    return numpy.dot(X, v)


def centerColumns(X) :
    '''returns X - mean(X), where the mean is taken over the columns of X'''
    
    m = numpy.mean(X)

    n = numpy.shape(X)[0]
    
    return X - numpy.resize(m, (n,len(m)))


def centerRows(X) :

    return numpy.transpose(centerColumns(numpy.transpose(X)))


def standardizeColumns(X) :
    '''returns (X - mean(X)) / std(X) '''

    m = numpy.mean(X)
    std = numpy.std(X)

    n = numpy.shape(X)[0]

    return (X - numpy.resize(m, (n,len(m))))/numpy.resize(std, (n,len(std)))
    
def standardizeRows(X) :
    '''returns (X - mean(X)) / std(X) '''

    return numpy.transpose(standardizeColumns(numpy.transpose(X)))
    

def maxvar(X, numVariables) :
    '''returns the numVariables variables with the highest variance'''
    
    s = numpy.std(X)
    I = numpy.argsort(s)
    
    Xout = numpy.take(X, I[-numVariables:], 1)

    return Xout


def dmat(X) :
    '''returns the Euclidean distance-squared matrix'''

    K = numpy.matrixmultiply (X, numpy.transpose (X))
    n = numpy.shape(K)[0]
    D = numpy.zeros((n,n), numpy.float)

    for i in range(1, n-1) :
        for j in range(i+1, n) :
            D[i,j] = K[i,i] - 2 * K[i,j] + K[j,j]
            D[j,i] = D[i,j]
    
    return D


def norm2(x) :
    '''return the 2-norm of a vector given as a list or numpy array'''
    
    x = numpy.asarray(x)
    
    return math.sqrt(numpy.sum(x*x))


def normalizeNorm(X) :
    '''normalize each row of X to unit vectors'''

    (numRows, numCols) = numpy.shape(X)
    Xnorm = numpy.zeros((numRows, numCols), numpy.float)
    
    for i in range(numRows) :
        Xnorm[i] = X[i] / norm2(X[i])

    return Xnorm
    
class Correlator (object) :

    def __init__(self, data) :

        if type(data) == type('') :
            print 'file name:', data            
            data = datafunc.PyVectorDataSet(data, idColumn = 0, headerRow = True, hint = 'csv')

        self.data = data
        self.idDict = misc.list2dict(data.labels.patternID,
                                     range(len(data)))

        print numpy.shape(data.X)
        self.mean = numpy.mean(data.X, 1)
        self.std = std(data.X, 1)
        eps = 1e-5
        I = numpy.nonzero(numpy.less(self.std, eps))[0]
        print 'num zeros:',len(I)
        numpy.put(self.std, I, 1)
        
        self.numCorrelations = 10000
        correlations = numpy.zeros(self.numCorrelations, numpy.float)
        
        for i in range(self.numCorrelations) :
            i1 = random.randrange(0, len(data))
            i2 = random.randrange(0, len(data))
            correlations[i] = self._corrcoef(i1, i2)
        self.meanCorrelation = numpy.mean(correlations)
        self.numCorrelations = 1000        

    def corrcoef(self, id1, id2) :

        if id1 == id2 : return 1.0
        if type(id1) == type(1) :
            return self._corrcoef(id1, id2)
        if id1 not in self.idDict and id2 not in self.idDict :
            return self.meanCorrelation
        if id1 in self.idDict and id2 in self.idDict :
            return self._corrcoef(self.idDict[id1], self.idDict[id2])
        else :
            # we want to assume that id1 not in data:
            if id2 not in self.idDict :
                id1,id2 = id2,id1
            i2 = self.idDict[id2]
            correlations = numpy.zeros(self.numCorrelations, numpy.float)
            for i in range(self.numCorrelations) :
                i1 = random.randrange(0, len(self.data))
                correlations[i] = self._corrcoef(i1, i2)
            return numpy.mean(correlations)

    def _corrcoef(self, i1, i2) :
        
        return numpy.dot(self.data.X[i1] - self.mean[i1],
                           self.data.X[i2] - self.mean[i2]) / \
                           (len(self.data.X[i1]) * self.std[i1] * self.std[i2])
    
    
def corrcoef2(X) :
    '''compute the correlation between the rows of the matrix X
    more space efficient than numpy version'''
    
    (n,d) = numpy.shape(X)

    m = numpy.mean(X, 1)
    std = numpy.std(X, 1)

    K = numpy.ones((n,n), numpy.float)

    for i in range(0, n - 1) :
        for j in range(i + 1, n) :
            K[i][j] = numpy.dot(X[i] - m[i], X[j] - m[i]) / (d * std[i] * std[j])
            K[j][i] = K[i][j]

    return K

def std(m,axis=0):
    """std(m,axis=0) returns the standard deviation along the given 
    dimension of m.  The result is unbiased with division by N-1.
    If m is of integer type returns a floating point answer.
    """
    x = numpy.asarray(m)
    n = float(x.shape[axis])
    mx = numpy.asarray(numpy.mean(x,axis))
    if axis < 0:
        axis = len(x.shape) + axis
    mx.shape = mx.shape[:axis] + (1,) + mx.shape[axis:]
    x = x - mx
    return numpy.sqrt(numpy.add.reduce(x*x,axis)/(n))

def corrcoef(X) :

    (n,d) = numpy.shape(X)

    Xn = standardizeRows(X)

    return numpy.dot(Xn, numpy.transpose(Xn)) / (d - 1)

def corrcoefij(X, i, j) :

    (n,d) = numpy.shape(X)

    m = numpy.mean(X, 1)
    std = numpy.std(X, 1)


    return numpy.dot(X[i] - m[i], X[j] - m[i]) / (d * std[i] * std[j])


class Standardizer (PyMLobject) :
    """
    class for performing feature normalization

    For each feature the Standardizer subtracts the feature's mean
    and divides by its standard deviation
    
    this rescaling is composed of two operations:
    
      1.  ``centering`` -- subtract from a feature its mean value;
          this is referred to as 'translation'; the translation attribute
          gives the value with which to translate each feature
      2.  ``scaling`` -- divide a feature by a scale, e.g. its standard deviation;
          the 'scale' attribute gives the value with which to scale each feature

    the 'train' method of the class computes the translation and scaling
    factors, and performs normalization of the training data
    the 'test' method uses values computed on the training data to normalize
    test data.

    :Keywords:
      - `skip_sparse` - whether to skip attributes that are determined to be sprase [True]
      - `sparsity_ratio` - sparsity is determined by the fraction of nonzero values.  When
        when the value exceeds the threshold, and skip_sparse is True, the attribute will be
        normalized.  Default value: 0.2
      - `translate` - whether to translate the data [True]
      - `rescale` - whether to rescale the data [True]
    
    **caveat:**
    Beware of performing training multiple times on the same dataset:
    if a dataset has already been standardized, re-standardization
    will recompute mean and standard deviation, which will be approximately
    0 and 1 for each feature; subsequent application on test data will
    have no effect. Because of this an exception is raised if the user
    attempts to re-train an already trained Rescale object.
    """

    attributes = {'skip_sparse' : True,
    'sparsity_ratio' : 0.2,
    'translate' : True,
    'rescale' : True,
    'translation' : None,
    'scale' : None}

    def __init__(self, **args) :

        PyMLobject.__init__(self, **args)

    def train(self, data, **args) :

        if self.translation is not None or self.scale is not None :
            raise ValueError, 'object already trained'
        if self.translate :
            self.translation = numpy.array( data.mean() )
        if self.rescale :
            self.scale = numpy.array( data.std() )
            # need to avoid division by 0, so
            # scales that are equal to 0 are replaced with a value of 1
            eps = 1e-5
            I = numpy.nonzero(numpy.less(self.scale, eps))[0]
            numpy.put(self.scale, I, 1)
            # checking for nan:
            for i in range(len(self.scale)) :
                if numpy.isnan(self.scale[i]) :
                    self.scale[i] = 1
            if self.skip_sparse :
                # locate sparse features and set their translation to 0 and rescale value to 1
                feature_counts = featsel.featureCount(data)
                for i in range(len(feature_counts)) :
                    if feature_counts[i] / len(data) < self.sparsity_ratio :
                        self.translation[i] = 0
                        self.scale[i] = 1
        self.preproc(data)

    def preproc(self, data) :

        if self.translate :
            data.translate(self.translation)
        if self.rescale :
            data.scale(1.0 / self.scale)

    def test(self, data, **args) :

        self.preproc(data)
    


from PyML.containers import labels
from PyML.classifiers import svm
from PyML.containers import ker
from PyML.utils import myio, misc
from PyML.evaluators import roc

import numpy
import random

'''classes for performing feature selection'''

__docformat__ = "restructuredtext en"


class FeatureSelector (object) :

    '''api for feature selection objects'''

    type = 'featureSelector'

    def select(self, data, **args) :
        """
        invokes ``selectFeatures`` to find predictive features and eliminates
        the rest of the features from the input dataset
        """
        
        features = self.selectFeatures(data, **args)
        #print '*** number of features: *** ', len(features)
        data.keepFeatures(features)
    
    def selectFeatures(self, data, *options, **args) :
        """
        :Returns:
          a list of predictive features
        """
        raise NotImplementedError

    def score(self, data, **args) :
        """
        :Returns:
          a score for each feature in the input dataset
        """
        raise NotImplementedError

    def rank(self, data, **args) :
        """
        :Returns:
          a ranking of the features in the dataset by converting the scores
          to ranks
        """
        scores = self.score(data, **args)

        return weights2ranks(scores, data)

    def test(self, data, *options, **args) :

        pass

    train = select

class OneAgainstRestSelect (FeatureSelector) :
    '''Use a two-class feature selection method for multi-class problem
    by doing feature selection in a one-against-the-rest manner, and
    returns the union of all the features selected.
    
    Construction::

      OneAgainstRestSelect(featureSelector) -- featureSelector is either
      a OneAgainstRestSelect object for copy construction, or a featureSelector
      object
    '''
    
    def __init__(self, featureSelector) :

        if (not hasattr(featureSelector, 'type') or
            featureSelector.type != 'featureSelector') :
            raise ValueError, 'need a feature selector as input'

        if featureSelector.__class__ == self.__class__ :
            self.featureSelector = featureSelector.featureSelector.__class__(
                featureSelector.featureSelector)
        else :
            self.featureSelector = featureSelector.__class__(featureSelector)

    def selectFeatures(self, data, **args) :

        labels = data.labels

        features = []
        for k in range(data.labels.numClasses) :
            data2 = labels.oneAgainstRest(data, k)
            features2 = self.featureSelector.selectFeatures(data2)
            features = misc.union(features, features2)
            data.attachLabels(labels)

        return features

    
class RFE (FeatureSelector) :

    '''
    RFE (Recursive Feature Elimination) uses the vector *w* of an SVM for
    feature selection.

    The method alternates between training a linear SVM and removing the features
    with the smallest value of the weight vector.

    You can either choose the number of features or let RFE choose the number
    of features automatically; this is chosen as the minimal number of features
    such that the number of support vectors is within one standard deviation
    from the minimum number of support vectors.

    Reference:
    
      I. Guyon and J. Weston and S. Barnhill and V. Vapnik
      Gene selection for cancer classification using support vector machines.
      Machine Learning 46:389-422, 2002.

    '''

    def initialize(self, data) :

        self.data = data.__class__(data, deepcopy = 1)
        if self.selectNumFeatures :
            self.featureLists = [data.featureID]
        #self.features = range(data.numFeatures)

        #self.featureLists = []
        self.wList = []
        self.numSV = []
        
    def __init__(self, arg = None, **settings) :

        """
        :Keywords:
          - `targetNum` - perform backward elimination until this many features are
            left
          - `mode` - values - 'byFraction' or 'byNum' (default = 'byFraction')
          - `numToEliminate` - specifies the number of features to eliminate at each
            iteration in the byNum mode
          - `fractionToEliminate` - the fraction of features to eliminate at each
            iteration in the byFraction mode (default = 0.1)
          - `autoSelect` [False] - whether the number of features should be chosen
            automatically
          - `useScore` - whether to modulate the vector w by the golub coefficient
            as in RSVM

        """

        self.selectNumFeatures = True
        self.fractionToEliminate = 0.1
        self.numToEliminate = 10
        self.mode = 'byFraction'  #values: byFraction or byNumber
        self.numFeatures = 20
        self.featureScore = FeatureScore('golub')
        self.useScore = False
        self.rankFeatures = False
        
        if arg is None :
            self.svm = svm.SVM()
        elif arg.__class__ == self.__class__ :
            other = arg
            self.fractionToEliminate = other.fractionToEliminate
            self.numToEliminate = other.numToEliminate
            self.mode = other.mode
            self.numFeatures = other.numFeatures
            self.selectNumFeatures = other.selectNumFeatures
            self.useScore = other.useScore
            self.svm = other.svm.__class__(other.svm)
        elif arg.__class__.__name__ == 'SVM' :
            self.svm = arg.__class__(arg)
        else :
            raise ValueError, 'unknown type of argument for RFE ' + str(arg)

        if 'mode' in settings :
            self.mode = settings['mode']
        if 'numToEliminate' in settings :
            self.numToEliminate = settings['numToEliminate']
        if 'numFeatures' in settings :
            self.numFeatures = settings['numFeatures']
        if 'fractionToEliminate' in settings :
            self.fractionToEliminate = settings['fractionToEliminate']
        if 'autoSelect' in settings :
            self.selectNumFeatures = settings['autoSelect']
        if 'useScore' in settings :
            self.useScore = settings['useScore']

    def __repr__(self) :
        rep = '<' + self.__class__.__name__ + ' instance>\n'
        rep += 'mode: ' + self.mode + '\n'
        if self.mode == "byNum" :
            rep += 'number of features to eliminate each iteration : %d\n' \
                   % self.numToEliminate
        elif self.mode == "byFraction" :
            rep += 'Fraction to eliminate each iteration : %f\n' \
                   % self.fractionToEliminate
        rep += 'target number of features : %d\n' % self.numFeatures
        rep += 'automatic selection of the number of features : %d' % \
               self.selectNumFeatures


        return rep

    def __iter__(self) :

        return self
        
    def getFeatures(self, w, numFeatures) :

        if self.mode == 'byNumber' :
            numToElim = min(self.numToEliminate,
                            numFeatures - self.numFeatures)
        elif self.mode == 'byFraction' :
            numToElim = min(int(self.fractionToEliminate * len(w)),
                            numFeatures - self.numFeatures)
        else :
            raise ValueError, 'invalid elimination mode'

        if numToElim == 0: numToElim = 1
        #print 'numFeaturesToEliminate: ', numToElim
        
        if type(w) == type({}) :
            w2 = numpy.zeros(numFeatures, numpy.float)
            for wKey in w.keys():
                w2[wKey] = w[wKey]
            w = w2
            
        w = numpy.absolute(w)

        if self.useScore :
            w = w * self.featureScore.score(self.data)

        numZero = numpy.sum(numpy.equal(w, 0))
        if numZero > numToElim : numToElim = numZero
        
        I = list(numpy.argsort(w))
        featuresToEliminate = I[:numToElim]

        self.features = I[numToElim:]
        self.w = w
        
        return featuresToEliminate


    def next(self) :

        data = self.data

        if data.numFeatures <= self.numFeatures :
            raise StopIteration

        self.svm.train(data)

        #self.wList.append(self.svm.model.w)
        self.numSV.append(self.svm.model.numSV)
        
        featuresToEliminate = self.getFeatures(self.svm.model.warray,
                                               data.numFeatures)
        #print featuresToEliminate, len(featuresToEliminate)
        if self.rankFeatures :
            if len(self.weights) == 0 :
                maxWeight = 0
            else :
                maxWeight = max(self.weights.values())
            for feature in featuresToEliminate :
                self.weights[data.featureID[feature]] = self.w[feature] + maxWeight

        print 'eliminating:', featuresToEliminate
        data.eliminateFeatures(featuresToEliminate)
        #print '** numFeatures: ', data.numFeatures
        
        if self.selectNumFeatures :
            self.featureLists.append(data.featureID)
        

    def run(self, data, **args) :

        if data.labels.numClasses != 2 :
            raise ValueError, 'RFE supports only two class problems'
        
        self.initialize(data)
        features = data.featureID[:]

        rfeIter = iter(self)
        for f in rfeIter : pass

        if self.selectNumFeatures :
            minNumSV = len(self.data) + 1
            for i in range(len(self.numSV)) :
                if self.numSV[i] < minNumSV :
                    minNumSV = self.numSV[i]
                    features = self.featureLists[i]
            #print features
            self.features = data.featureNames2IDs(features)


    def selectFeatures(self, data, **args):

        self.run(data, **args)
        
        return self.features

    def rank(self, data, **args):

        self.rankFeatures = True
        self.weights = {}

        self.run(data, **args)

        # add the weights from the features that remain:
        if len(self.weights) == 0 :
            maxWeight = 0
        else :
            maxWeight = max(self.weights.values())
        for feature in range(self.data.numFeatures) :
            self.weights[self.data.featureID[feature]] = self.w[feature] + maxWeight

        weights = [self.weights[data.featureID[i]]
                   for i in range(data.numFeatures)]
        I = numpy.argsort(weights)

        #ranks = [data.featureID[i] for i in I]

        return weights2ranks(weights, data)


class MultiplicativeUpdate (FeatureSelector) :
    '''Multiplicative update uses the vector w of an SVM to do feature selection.
    At each iteration an svm is trained and the data is multiplied by the
    weight vector of the classifier.
    
    Reference:
    
    J. Weston, A. Elisseeff, M. Tipping and B. Scholkopf.
    Use of the zero norm with linear models and kernel methods.
    JMLR special Issue on Variable and Feature selection, 2002.
    '''

    
    def __init__(self, arg = None, **settings) :

        self.eps = 0.01
        self.rankFeatures = False
        
        if arg.__class__ == self.__class__ :
            other = arg
            self.eps = other.eps
            self.rankFeatures = other.rankFeatures
        elif arg.__class__.__name__ == 'SVM' :
            self.svm = arg.__class__(arg)

        if 'eps' in settings :
            self.eps = settings['eps']


    def __repr__(self) :
        rep = '<' + self.__class__.__name__ + ' instance>\n'
        rep += 'epsilon : %d\n' % self.eps

        return rep
        
    def __iter__(self) :

        return self

    def initialize(self, data) :

        self.scaleData = data.__class__(data, deepcopy = True)
        if not linearlySeparable (data) :
            print 'not linearly separable!!!!!!!!!!!!!!!!!!!!!!'
            self.svm = svm.SVM(ker.LinearRidge())
        else :
            self.svm = svm.SVM()
            print 'linearly separable**************************'
        self.svm.C = 1000
	

    def next(self) :

	data = self.scaleData
	self.svm.train(data)
	#w = self.svm.model.w
	w = self.svm.model.warray
	if self.svm.kernel.__class__.__name__ == "LinearRidge" :
	    wRidge = 0.0
	    for i in range(self.svm.model.numSV) :
		wRidge += self.svm.model.alpha[i] * \
			  self.svm.ridge[self.svm.model.svID[i]]
	    wRidge = abs(wRidge)
	    for i in range(len(data)) :
		self.svm.ridge[i] *= wRidge
            
        data.scale(w)
	self.w = w
	wc = numpy.compress(numpy.greater(w, 1e-3), w)
	
	if numpy.allclose(wc, numpy.ones(len(wc), numpy.float), 0.3) :
	    raise StopIteration


    def selectFeatures(self, data, *options, **args):
        '''XXX for multi-class -- do one against the rest
        and use the absolute value of the average/maximum value of w to rescale
        multi-class
        '''

        if data.labels.numClasses != 2 :
            raise ValueError, 'MU supports only two class problems'

        self.initialize(data)

        muIter = iter(self)
        for f in muIter : pass

        featuresToKeep = numpy.nonzero(numpy.greater(self.w, 1e-3))[0]

        return featuresToKeep


class Random (FeatureSelector) :
    '''
    A feature selection method that keeps a random set of features

    Construction::

      Random(numFeatures)
    '''

    def __init__(self, arg1, *options, **settings) :

        if arg1.__class__ == self.__class__ :
            other = arg1
            self.numFeatures = other.numFeatures
        elif type(arg1) == type(1) :
            self.numFeatures = arg1
        else :
            raise ValueError, 'bad argument for Random constructor'
                  
    def __repr__(self) :
        rep = '<' + self.__class__.__name__ + ' instance>\n'
        rep += 'number of features to keep : %d\n' % self.numFeatures
        
        return rep

    def selectFeatures(self, data, *options, **args) :

        if data.numFeatures <= self.numFeatures :
            return

        return misc.randsubset(data.numFeatures, self.numFeatures)
    
        
class Filter (FeatureSelector) :
    '''
    A simple feature selection method that filters features according
    to a feature score.
    It uses a feature score (instance of FeatureScore) to eliminate
    features in one of three possible modes:
    
    - keep a specified number of features [default]
    - eliminate all features whose score is below some threshold
    - eliminate all features whose score is a certain number of standard deviations
      above that obtained using random labels
    '''

    def __init__(self, arg1, *options, **settings) :
        """
        :Keywords:
          - `numFeatures` - keep ``numFeatures`` features with the highest score
          - `threshold` - keep all features with score above the threshold
          - `sigma` - keep features whose score is above the average by this many
            standard deviations
        """
        self.sigma = 2.5
        if arg1.__class__ == self.__class__ :
            other = arg1
            self.featureScore = other.featureScore.__class__(other.featureScore)
            self.numFeatures = other.numFeatures
            self.mode = other.mode
            self.numRand = other.numRand
            self.sigma = other.sigma
            try :
                self.threshold = other.threshold
            except :
                pass
            try :
                self.significance = other.significance
            except :
                pass
            try :
                self.numFeatures = other.numFeatures
            except :
                pass
        elif hasattr(arg1, 'score') :
            self.featureScore = arg1
            self.mode = "byNum"
            self.numFeatures = 20
            self.numRand = 20
            if 'numFeatures' in settings :
                self.numFeatures = settings['numFeatures']
                self.mode = "byNum"
            if 'sigma' in settings :
                self.sigma = settings['sigma']
                self.mode = "bySignificance"
            if 'threshold' in settings :
                self.threshold = settings['threshold']
                self.mode = "byThreshold"
        else :
            raise ValueError, 'bad argument for Filter constructor'

    def __repr__(self) :
        rep = '<' + self.__class__.__name__ + ' instance>\n'
        rep += 'mode: ' + self.mode + '\n'
        if self.mode == "byNum" :
            rep += 'number of features to keep : %d\n' % self.numFeatures
        elif self.mode == "bySignificance" :
            rep += 'sigma : %f\n' \
                   % self.sigma

        elif self.mode == "byThreshold" :
            rep += 'score threshold for keeping features : %f\n' % self.threshold
        rep += self.featureScore.__repr__()
        
        return rep

    def selectFeatures(self, data, targetClass=None, otherClass = None, *options, **args) :

        s = self.featureScore.score(data, targetClass, otherClass, **args) 

        if self.mode == "byNum" :
            featuresToEliminate = numpy.argsort(s)\
                                  [:data.numFeatures - self.numFeatures]
        elif self.mode == "byThreshold" :
            featuresToEliminate = numpy.nonzero(numpy.less(s, self.threshold))[0]
        elif self.mode == "bySignificance" :
            t = self.significanceThreshold(data)
            self.thresholds = t
            featuresToEliminate = numpy.nonzero(numpy.less(s, t))[0]
        else :
            raise ValueError, 'unknown elimination mode in filter'

        return misc.setminus(range(data.numFeatures), featuresToEliminate)


    def significanceThreshold(self, data) :

        s = numpy.zeros((self.numRand,data.numFeatures), numpy.float)
        
        for i in range(self.numRand) :
            Y = labels.randomLabels(data.labels.Y)
            s[i] = self.featureScore.score(data, Y = Y)

        #t = [misc.inverseCumulative(s[:,j], self.significance)
        #     for j in range(data.numFeatures)]
        #print t
        #print max(t)
        t = s.mean() + self.sigma * s.std()
        return t


def parseArgs(data, targetClass, otherClass = None, **args) :
    '''parse arguments for a feature scoring function'''

    if 'feature' in args :
        feature = args['feature']
    else :
        feature = None
    if 'Y' in args :
        Y = args['Y']
        if otherClass is None :
            otherI = numpy.nonzero(numpy.not_equal(Y, targetClass))[0]
        else :
            otherI = numpy.nonzero(numpy.equal(Y, otherClass))[0]
        targetClassSize = numpy.sum(numpy.equal(Y, targetClass))
    else :
        Y = None
        if otherClass is None :
            otherI = numpy.nonzero(numpy.not_equal(data.labels.Y, targetClass))[0]
        else :
            otherI = data.labels.classes[otherClass]
        targetClassSize = len(data.labels.classes[targetClass])
    
    otherClassSize = len(otherI)

    return Y, targetClassSize, otherClassSize, otherI, feature


def singleFeatureSuccRate(data, targetClass, otherClass = None, **args) :

    Y, targetClassSize, otherClassSize, otherI, feature = parseArgs(
        data, targetClass, otherClass, **args)

    if Y is None : Y = data.labels.Y
    if data.__class__.__name__ != 'DataSet' :
        raise ValueError, 'data should be of type DataSet'
    
    Xsort = numpy.sort(data.X, 0)
    d = data.numFeatures
    n = len(data)
    Isort = numpy.argsort(data.X, 0)
    succRate = numpy.zeros(d, numpy.float)
    threshold = numpy.zeros(d, numpy.float)    
    num1 = numpy.sum(numpy.equal(Y, 1))
    num0 = n - num1
    
    for i in range(d) :
        succRate[i] = 0
        num0below = 0
        num1below = 0
        for j in range(0, n - 1) :
            if Y[Isort[j][i]] == 1 :
                num1below += 1
            else :
                num0below += 1
            num0above = num0 - num0below
            num1above = num1 - num1below
            currSuccRate = float(max(num0above + num1below, num0below + num1above)) / \
                           float(n)
            if currSuccRate > succRate[i] :
                succRate[i] = currSuccRate
                threshold[i] = (Xsort[j][i] + Xsort[j + 1][i]) / 2

    return succRate,threshold
                
        
def predictivity(data, targetClass, otherClass = None, **args) :
    
    '''A feature score for discrete data; the score for feature i is:
    s_i = P(Fi | C1) - P(Fi | C2),
    where P(Fi | C) is the estimated probability of Feature i being nonzero given
    the class variable
    This is estimated as:
    s_i = # of patterns in target class that have feature i /
          no. of patterns in target class
          -
          # of patterns in other class that have feature i /
          no. of patterns in other class
    '''
    
    Y, targetClassSize, otherClassSize, otherI, feature = parseArgs(
        data, targetClass, otherClass, **args)
                                                           

    s1 = numpy.array(featureCount(data, targetClass=targetClass, Y=Y,
                                  feature=feature)) / float(targetClassSize)

    s2 = numpy.array(featureCount(data, I = otherI, Y=Y,
                                  feature=feature)) / float(otherClassSize)
    
    return (s1 - s2) 


def countDiff(data, targetClass, otherClass = None, **args) :
    '''A feature score for discrete data; the score for feature i is:
    s_i = (#(Fi | C ) - #(Fi | not C)) / #(Fi | C)
    '''

    Y, targetClassSize, otherClassSize, otherI, feature = parseArgs(
        data, targetClass, otherClass, **args)

    s1 = featureCount(data, targetClass=targetClass, Y=Y,
                      feature=feature) 

    s2 = featureCount(data, I = otherI, Y=Y,
                      feature=feature)

    s = (s1 - s2) / float(targetClassSize)
    
    return s


def sensitivity(data, targetClass, otherClass = None, **args) :
    '''A feature score for discrete data
    (alternatively, with a threshold it could be used for continuous data)
    s_i = #(Fi | C) / #(C)
    '''

    Y, targetClassSize, otherClassSize, otherI, feature = parseArgs(
        data, targetClass, otherClass, **args)

    return (featureCount(data, targetClass=targetClass, Y=Y, feature=feature) /
            float(targetClassSize))



def ppv(data, targetClass, otherClass = None, **args) :
    '''A feature score for discrete data
    s_i = #(Fi | C) / #(Fi)
    '''

    Y, targetClassSize, otherClassSize, otherI, feature = parseArgs(
        data, targetClass, otherClass, **args)

    s1 = featureCount(data, targetClass=targetClass, Y=Y, feature=feature)

    s2 = featureCount(data, feature = feature)

    numpy.putmask(s2, numpy.equal(s2, 0), 1)

    if type(s1) == type(1) :
        return float(s1) / float(s2)
    else :
        return numpy.array(s1, numpy.float)/s2

def ppvThreshold(data, targetClass, otherClass = None, **args) :
    '''A feature score for discrete data
    s_i = #(Fi | C) / #(Fi) if #(Fi | C) > threshold and 0 otherwise
    '''

    Y, targetClassSize, otherClassSize, otherI, feature = parseArgs(
        data, targetClass, otherClass, **args)
    if 'threshold' in args :
        threshold = args['threshold']
    else :
        threshold = 2

    s1 = featureCount(data, targetClass=targetClass, Y=Y, feature=feature)

    numpy.putmask(s1, numpy.less_equal(s1, threshold), 0)
    
    s2 = featureCount(data, feature = feature)
    # avoid division by 0 :
    numpy.putmask(s2, numpy.equal(s2, 0), 1)

    if type(s1) == type(1) :
        return float(s1) / float(s2)
    else :
        return numpy.array(s1, numpy.float)/s2


def specificity(data, targetClass, otherClass = None, **args) :
    '''A feature score for discrete data
    s_i = #(Fi | C) / #(Fi)

    or perhaps: 1 - #(Fi | not C) / #(not C)
    
    '''

    Y, targetClassSize, otherClassSize, otherI, feature = parseArgs(
        data, targetClass, otherClass, **args)

    s1 = featureCount(data, targetClass=targetClass, Y=Y, feature=feature)

    s2 = featureCount(data, feature = feature)

    numpy.putmask(s2, numpy.equal(s2, 0), 1)

    if type(s1) == type(1) :
        return float(s1) / float(s2)
    else :
        return numpy.array(s1, numpy.float)/s2


def usefullness(data, targetClass, otherClass = None, **args) :
    '''A feature score for discrete data
    optional arguments:
    threshold
    fraction
    '''

    if 'threshold' in args :
        threshold = args['threshold']
    else :
        threshold = 5
    if 'fraction' in args :
        fraction = args['fraction']
    else :
        fraction = 0.0

    Y, targetClassSize, otherClassSize, otherI, feature = parseArgs(
        data, targetClass, otherClass, **args)

    threshold = max(threshold, fraction * float(targetClassSize))        

    s1 = featureCount(data, targetClass=targetClass, Y=Y, feature=feature)

    s2 = featureCount(data, I = otherI, Y=Y,
                      feature=feature) / float(otherClassSize)

    s2 = 1 - s2

    numpy.putmask(s2, numpy.less(s1, threshold), 0.0)

    return s2


def abundance(data, targetClass, otherClass = None, **args) :

    '''Fraction of patterns that have a feature: A(F,C) = #(F | C) \ #(C)'''

    Y, targetClassSize, otherClassSize, otherI, feature = parseArgs(
        data, targetClass, otherClass, **args)    

    s = featureCount(data, targetClass=targetClass, Y=Y, feature=feature) / \
        float(targetClassSize)

    return s



def oddsRatio(data, targetClass, otherClass = None, **args) :


    Y, targetClassSize, otherClassSize, otherI, feature = parseArgs(
        data, targetClass, otherClass, **args)

    count1 = numpy.array(featureCount(data, targetClass=targetClass, Y=Y,
                                        feature=feature), numpy.float)
    count2 = numpy.array(featureCount(data, I=otherI, Y=Y,
                                        feature=feature), numpy.float)

    pseudoCount1 = 1.0 / float(targetClassSize)
    pseudoCount2 = 1.0 / float(otherClassSize)
    numpy.putmask(count1, numpy.equal(count1, 0), pseudoCount1)
    numpy.putmask(count2, numpy.equal(count2, 0), pseudoCount2)
    numpy.putmask(count1, numpy.equal(count1, targetClassSize),
                    targetClassSize - pseudoCount1)
    numpy.putmask(count2, numpy.equal(count2, len(otherI)),
                    len(otherI) - pseudoCount2)

    
    s = (count1 * (otherClassSize - count2)) / (count2 * (targetClassSize - count1))

    return s

def logOddsRatio(data, targetClass, otherClass = None, **args) :

    return numpy.log(oddsRatio(data, targetClass, otherClass, **args))



def relief (data) :

    if type(data.X[0]) == type({}) :
        raise valueError, "Wrong type of dataset"            
    if data.labels.numClasses != 2 :
        raise valueError, 'not a two class problem'

    K = numpy.dot (data.X, numpy.transpose (data.X))

    w = numpy.zeros(data.numFeatures, numpy.float)
    for i in range(len(data)) :
        bestInClass = 0
        simInClass = -1e10
        bestOutOfClass = 0
        simOutOfClass = -1e10
        for j in range(len(data)) :
            if j == i : continue
            if data.labels.Y[i] == data.labels.Y[j] :
                if K[i][j] > simInClass :
                    bestInClass = j
                    simInClass = K[i][j]
            else :
                if K[i][j] > simOutOfClass :
                    bestOutOfClass = j
                    simOutOfClass = K[i][j]
        w += data.X[bestInClass] - data.X[bestOutOfClass]

    return w / len(data)

    
def golub(data, targetClass, otherClass, **args) :
    '''The Golub feature score:
    s = (mu1 - mu2) / sqrt(sigma1^2 + sigma2^2)
    '''

    if 'Y' in args :
        Y = args['Y']
        targetClassSize = numpy.sum(numpy.equal(Y, targetClass))
        otherClassSize = numpy.sum(numpy.equal(Y, otherClass))        
    else :
        Y = None
        targetClassSize = data.labels.classSize[targetClass] 
        otherClassSize = data.labels.classSize[otherClass]
    
    m1 = numpy.array(featureMean(data, targetClass, Y))
    m2 = numpy.array(featureMean(data, otherClass, Y))
    s1 = numpy.array(featureStd(data, targetClass, Y))
    s2 = numpy.array(featureStd(data, otherClass, Y))

    s = numpy.sqrt(s1**2 + s2**2)
    m = (m1 + m2) / 2.0

    # perfect features will have s[i] = 0, so need to take care of that:
    numpy.putmask(s, numpy.equal(s, 0), m)
    # features that are zero will still have s[i] = 0 so :
    numpy.putmask(s, numpy.equal(s, 0) ,1)
    
    g = (m1 - m2) / s
    
    return g

def succ(data, targetClass, otherClass, **args) :
    """the score of feature j is the success rate of a classifier that
    classifies into the target class all points whose value of the feature
    are higher than some threshold (linear 1-d classifier).
    """
    Y = data.labels.Y
    numPos = float(data.labels.classSize[targetClass])
    numNeg = len(data) - numPos
    s = numpy.zeros(data.numFeatures, numpy.float_)
    values = numpy.zeros(data.numFeatures, numpy.float_)
    balanced = False
    if 'balanced' in args :
        balanced = args['balanced']
    #negFrac = float(numNeg) / float(len(data))
    #posFrac = float(numPos) / float(len(data))
    for j in range(data.numFeatures) :
        feat = data.getFeature(j)
        I = numpy.argsort(feat)
        feat = numpy.sort(feat)
        posBelow = 0
        negBelow = 0
        for i in range(len(data)) :
            if Y[I[i]] == targetClass :
                posBelow += 1
            else :
                negBelow += 1
            # the following if statement takes into account
            # discrete data. in that case the decision is made only
            # when the feature changes its value
            if i < len(data)-1 and feat[i] != feat[i + 1] :
                if balanced :
                    succRate = max(posBelow / numPos + (numNeg - negBelow) / numNeg,
                                   (numPos - posBelow) / numPos + negBelow / numNeg)
                else :
                    succRate = max(posBelow + (numNeg - negBelow),
                                   (numPos - posBelow) + negBelow)
                if succRate > s[j] :
                    s[j] = succRate
                    values[j] = feat[i]
                               
    if not balanced : 
        s = s / len(data)
    else :
        s = s / 2.0

    if 'getValues' in args and args['getValues'] :
        return s,values
    else :
        return s

def balancedSucc(data, targetClass, otherClass, **args) :
    """the score of feature j is the success rate of a classifier that
    classifies into the target class all points whose value of the feature
    are higher than some threshold (linear 1-d classifier).
    """

    return succ(data, targetClass, otherClass, **{'balanced' : True})

def roc_score(data, targetClass, otherClass, **args) :

    rocN = None
    if 'rocN' in args :
        rocN = args['rocN']
    s = numpy.zeros(data.numFeatures, numpy.float_)
    for i in range(data.numFeatures) :
        featureValues = data.getFeature(i)
        auc = roc.roc(featureValues, data.labels.Y)[2]
        s[i] = max(auc, 1-auc)

    return s
        

def featureCount(data, *options, **args) :
    '''
    returns a vector where component i gives the number of patterns where
    feature i is nonzero
    INPUTS:
    data - a dataset
    targetClass - class for which to count (optional, default behavior is
    to look at all patterns)
    Y - alternative label vector (optional)
    feature - either a feature or list of features - counts the number of
    patterns for which the feature or list of features is non-zero
    I - a list of indices on which to do feature count
    OPTIONS:
    "complement" - look at the complement of the target class
    '''

    singleFeature = 0
    if 'feature' in args and args['feature'] is not None :
        feature = args['feature']
        singleFeature = 1
        featureCount = 0
    else :
        featureCount = numpy.zeros(data.numFeatures)

    if 'Y' in args and args['Y'] is not None :
        Y = args['Y']
    elif 'labels' in args :
        Y = args['labels'].Y
    elif data.labels.L is not None :
        Y = data.labels.Y

    if "targetClass" in args :
        targetClass = args['targetClass']
        if "complement" in options :
            I = numpy.nonzero(numpy.not_equal(Y, targetClass))[0]
        else :
            I = numpy.nonzero(numpy.equal(Y, targetClass))[0]
    else :
        I = range(len(data))

    if 'I' in args :
        I = args['I']

    if singleFeature :
        featureCount = data.featureCount(feature, I)
    else :
        featureCount = data.featureCounts(I)
            
    return featureCount


def featureMean(data, targetClass = None, Y = None) :
    '''returns a vector where component i is the mean of feature i
    INPUT:
    data - a dataset
    targetClass - class for which to take the mean (optional)
    Y - alternative label vector (optional)
    '''
    
    if targetClass is None :
        I = range(len(data))
    elif Y is None :
        I = numpy.nonzero(numpy.equal(data.labels.Y, targetClass))[0]
    else :
        I = numpy.nonzero(numpy.equal(Y, targetClass))[0]

    return data.mean(I)



def featureStd(data, targetClass = None, Y = None) :
    '''returns a vector where component i is the standard deviation of feature i
    INPUT:
    data - a dataset
    targetClass - class for which to take the mean (optional)
    Y - alternative label vector (optional)
    '''
    
    if targetClass == None :
        I = range(len(data))
    elif Y == None :
        I = numpy.nonzero(numpy.equal(data.labels.Y, targetClass))[0]
    else :
        I = numpy.nonzero(numpy.equal(Y, targetClass))[0]

    if len(I) == 0 :
        return numpy.zeros(data.numFeatures, numpy.float_)

    return data.std(I)


def eliminateSparseFeatures(data, threshold) :
    '''removes from the data features whose feature count is below a threshold
    data - a dataset
    threshold - number of occurrences of the feature below which it will be
    eliminated
    '''
    
    fCount = featureCount(data)

    below = numpy.nonzero(numpy.less(fCount, threshold))[0]
    data.eliminateFeatures(below)
    
    

def nonredundantFeatures(data, w = None) :
    '''Compute a set of nonredundant features for a 0/1 sparse dataset
    a feature is defined as redundant if there is another feature which has
    nonzero value for exactly the same patterns, and has a larger weight
    INPUT: a dataset and a list of weights for each feature in the data
    weights are optional.
    OUTPUT: a list of redundant features
    '''

    #data.featureView()

    bestFeature = {}
    featureWeight = {}

    for f in range(data.numFeatures) :
        if f % 100 == 0 :
            print f
        pattern = ''
        for i in range(len(data)) :
            if data.X[i].has_key(f) :
                pattern += '1'
            else :
                pattern += '0'
        if pattern in bestFeature :
            if w is not None :
                if featureWeight[pattern] < w[f] :
                    featureWeight[pattern] = w[f]
                    bestFeature[pattern] = f
        else :
            if w is not None :
                featureWeight[pattern] = w[f]
            bestFeature[pattern] = f

    nonredundant = bestFeature.values()

    return nonredundant


class FeatureScorer (object) :
    """base class for objects that have a 'score' function
    for scoring the features of a dataset
    """
    
    type = 'featureScorer'

    def score(self) :

        raise NotImplementedError
    
    train = score

    def test(self, data, *options, **args) :

        pass
    
class FeatureScore (FeatureSelector) :
    """
    A class for scoring the features of a dataset
    USAGE:
    construction:
    f = FeatureScore(scoreName, mode = modeValue)
    or using copy construction :
    f = FeatureScore(otherFeatureScore)
    scoreName is the type of filter; available filters are:
    "predictivity", "oddsRatio", "golub"
    mode is one of the following:
    oneAgainstRest (default)
    oneAgainstOne
    """

    scoreFuncs = {"predictivity" : predictivity,
                  "oddsRatio" : "oddsRatio", "logOddsRatio" : logOddsRatio,
                  "golub" : golub, "countDiff" : countDiff,
                  "usefullness" : usefullness, "abundance" : abundance,
                  "specificity" : specificity, "ppv" : ppv,
                  "ppvThreshold" : ppvThreshold,
                  "succ" : succ,
                  "balancedSucc" : balancedSucc,  "roc" : roc_score}
                   
    
    # multiClass tells whether a filter function handles multi-class data
    # otherwise, a feature is scored according to the maximum
    # pairwise score between classes
    
    multiClass = ["IG"]

    # asym tells whether a two-class filter function satisfies :
    # s(F,C1) = - s(F,C2)
    # for such functions a feature is scored as the absolute
    # value of the score when no class is given

    asym = ["predictivity", "logOddsRatio", "golub"]
    
    def __init__(self, arg1 = None, *options, **args) :

        self.mode = "oneAgainstOne"
        self.scoreName = "predictivity"
        self.scoreFunc = predictivity
        self.minClassSize = 5
        self.bothSides = True
        
        if arg1.__class__ == self.__class__ :
            other = arg1
            self.mode = other.mode
            self.scoreName = other.scoreName
            self.scoreFunc = other.scoreFunc
            self.bothSides = other.bothSides
        elif arg1.__class__ == ''.__class__ :
            scoreName = arg1
            if scoreName in self.scoreFuncs :
                self.scoreFunc = self.scoreFuncs[scoreName]
            else :
                raise ValueError, 'unknown filter name'
            self.scoreName = scoreName
        elif arg1.__class__.__base__.__name__ == 'FeatureScorer' :
            self.scoreFunc = arg1.score
            self.scoreName = ''

        if 'mode' in args :
            if args['mode'] == "oneAgainstRest" :
                self.mode = "oneAgainstRest"
        if 'minClassSize' in args :
            self.minClassSize = args['minClassSize']

        
    def __repr__(self) :
        rep = '<' + self.__class__.__name__ + ' instance>\n'
        rep += 'score name : ' + self.scoreName + '\n'
        rep += 'mode : ' + self.mode + '\n'
        
        return rep
    
        
        
    def score(self, data, *options, **args) :

        if 'targetClass' in args :
            targetClass = args['targetClass']
        else :
            targetClass = None
        if 'otherClass' in args :
            otherClass = args['otherClass']
        else :
            otherClass = None
            
        if (targetClass is not None and otherClass is not None) or (
            self.scoreName in self.multiClass) :
            return self.scoreFunc(data, targetClass, otherClass, **args)
        elif data.labels.numClasses == 2 :
            return self._score(data, **args)
        elif self.mode == "oneAgainstRest" :
            if targetClass is not None :
                labels = labels.oneAgainstRest(data.labels, targetClass)
                return self._score(data, 1, 0, Y=labels.Y)
            else :
                raise ValueError, 'need to specify a target class'
        elif self.mode == 'oneAgainstOne' :  
            return self.oneAgainstOne(data, targetClass, **args)

    train = score

    def _score(self, data, class1 = None, class2 = None, **args) :

        if class1 is None and class2 is None :
            class1 = 0
            class2 = 1
            
        if self.scoreName in self.asym or not self.bothSides :
            s = numpy.absolute(
                self.scoreFunc(data, class1, class2, **args))
        else :
            s = numpy.maximum(
                self.scoreFunc(data, class1, class2, **args),
                self.scoreFunc(data, class2, class1, **args))

        return s
    
    def oneAgainstOne(self, data, targetClass, **args) :
        '''XXXX change maximum into average or add this as another option'''

        if 'Y' in args :
            Y = args['Y']
            classSize = misc.count(Y)
        else :
            classSize = data.labels.classSize

        s = numpy.zeros(data.numFeatures, numpy.float_)

        if targetClass is None :
            for class1 in range(data.labels.numClasses - 1) :
                for class2 in range(class1 + 1, data.labels.numClasses) :
                    if (classSize[class1] > self.minClassSize and
                        classSize[class2] > self.minClassSize) :

                        t = self._score(data, class1, class2, **args)
                        s = numpy.maximum(s, t)
            
        else :
            for class2 in range(data.labels.numClasses) :
                if class2 != targetClass and classSize[class2] > self.minClassSize:
                    t = self._score(data, class1, class2, **args)
                    s = numpy.maximum(s, t)

        return s

class BackwardSelector (FeatureSelector) :

    def __init__(self, arg, **args) :

        self.measure = 'successRate'
        self.targetNumFeatures = 2
        if arg.__class__ == self.__class__ :
            self.measure = arg.measure
            self.targetNumFeatures = arg.targetNumfeatures
            self.classifier = arg.classifier.__class(arg.classifier)
        else :
            self.classifier = arg.__class__(arg)
        if 'targetNumFeatures' in args :
            self.targetNumFeatures = args['targetNumFeatures']
        if 'measure' in args :
            self.measure = args['measure']

    def selectFeatures(self, _data, *options, **args) :

        self.eliminated = []
        self.measures = []
        cvArgs = {}
        import re
        rocExp = re.compile(r"roc(?P<rocN>[0-9]+)area")
        match = rocExp.match(self.measure)
        if match is not None :
            measureStr = 'rocNarea'
            cvArgs['rocN'] = match.groupdict()['rocN']
        else :
            measureStr = self.measure
            
        data = _data.__class__(_data, deepcopy = True)
        for i in range(self.targetNumFeatures, _data.numFeatures) :
            maxScore = 0
            # loop over the CURRENT features
            for feature in range(data.numFeatures) :
                featureName = data.featureID[feature]
                data.eliminateFeatures([feature])
                res = self.classifier.stratifiedCV(data, **cvArgs)
                score = getattr(res, measureStr)
                if score > maxScore :
                    maxScore = score
                    bestFeatureName = featureName
                data = _data.__class__(_data, deepcopy = True)
                data.eliminateFeatures(data.featureNames2IDs(self.eliminated))
            data = _data.__class__(_data, deepcopy = True)
            self.eliminated.append(bestFeatureName)
            data.eliminateFeatures(data.featureNames2IDs(self.eliminated))
            self.measures.append(maxScore)

        return misc.setminus(range(_data.numFeatures),
                             _data.featureNames2IDs(self.eliminated))

def linearlySeparable (data) :
    '''returns 1 if data is linearly separable and 0 otherwise.
    More specifically, it trains a soft margin SVM and checks if all
    training points are correclty classified
    '''

    s = svm.SVM(C = 1000)
    s.train(data)
    r = s.test(data)
    r.computeStats()

    successRate = r.get('successRate')
    if successRate == 1 :
        return True
    else :
        return False
    
    
def extractNumFeatures(resultsFileName) :

    r = myio.load(resultsFileName)

    numFeatures = {}
    if type(r) == type({}) :
        info = misc.extractAttribute(r, 'foldInfo')
        for key in info :
            numFeat = []
            for lines in info[key] :
                for line in lines.split('\n') :
                    if line.find('number of features') == 0 :
                        numFeat.append(float(line.split(':')[1]))
            numFeatures[key] = numpy.average(numFeat)
    return numFeatures

def weights2ranks(weights, data) :

    if type(weights) == type({}) :
        weights = [weights[data.featureID[i]]
                   for i in range(data.numFeatures)]
    weights = numpy.array(weights)
    I = numpy.argsort(-weights)
    ranks = [data.featureID[i] for i in I]

    return ranks        

def featureReport(data, score = 'roc', targetClass = 1, otherClass = 0) :

    if score == 'roc' :
        s = roc(data, targetClass, otherClass)
    elif score == 'golub' :
        s = golub(data, targetClass, otherClass)
        
    for i in range(data.numFeatures) :
        print data.featureID[i], s[i]

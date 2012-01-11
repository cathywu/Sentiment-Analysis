import numpy
import random
import math
import os
import tempfile
import copy
import time

from PyML.utils import myio,misc
from PyML.evaluators import roc as roc_module

"""functionality for assessing classifier performance"""

__docformat__ = "restructuredtext en"

def scatter(r1, r2, statistic = 'roc', x1Label = '', x2Label= '',
            fileName = None, **args) :
    """
    a scatter plot for comparing the performance of two classifiers

    :Parameters:
      - `r1, r2` - both are either a list of Result classes, or a list of
        success rates / ROC scores
      - `statistic` - which measure of classifier success to plot
        values : 'roc', 'successRate', 'balancedSuccessRate'
	in order to specify parts of the roc curve you can use something like:
	'roc50' or 'roc0.1'

    :Keywords:
      - `title` - the title of the plot
    """

    if len(r1) != len(r2) :
        print 'unequal lengths for r1 and r2'
        if type(r1) != type({}) :
            raise ValueError, 'Cannot handle unequal length when it is not a dict'
        keys1 = r1.keys()
        keys2 = r2.keys()
        common = misc.intersect(keys1, keys2)
        r1new = {}
        r2new = {}
        for key in common :
            r1new[key] = r1[key]
            r2new[key] = r2[key]
        r1 = r1new
        r2 = r2new
        
    if type(r1) == type({}) and type(r2) == type({}) :
        I = r1.keys()
    else :
        I = range(len(r1))

    if (r1[I[0]].__class__.__name__ == 'Results' or
        r1[I[0]].__class__.__name__ == 'Container') :
        p1 = misc.extractAttribute(r1, statistic)
        p2 = misc.extractAttribute(r2, statistic)
    else :
        p1 = r1
        p2 = r2
        
    if type(p1) == type({}) :
        p1 = p1.values()
        p2 = p2.values()

    from matplotlib import pylab

    x = numpy.arange(0,1,0.01)
    pylab.plot(p1, p2, 'bo',x,x, '-k')
    pylab.xlabel(x1Label, fontsize = 18)
    pylab.ylabel(x2Label, fontsize = 18)
    if 'title' in args :
        pylab.title(args['title'], fontsize = 18)
    pylab.show()

    if fileName is not None :
        pylab.savefig(fileName)
	pylab.close()

def significance(r1, r2, statistic = 'roc') :
    """
    report the statistical significance of the difference in error rates
    of a series of classification results of two classifiers
    using the Wilcoxon signed rank test.

    Returns: pvalue, (median1, median2)
    where:
    pvalue - the pvalue of the two sided Wilcoxon signed rank test; to get
    the pvalue of a one sided test divide the pvalue by two.
    (median1, median2) - the median of the statistics of the inputs r1 and r2.

    :Parameters:
      - `r1, r2` - both are either a list of Result classes, or a list of success
        rates
      - `statistic` - which measure of classifier success to plot
        values : 'roc', 'successRate', 'balancedSuccessRate'
	in order to specify parts of the roc curve you can use something like:
	'roc50' or 'roc0.1'

    """

    if type(r1) != type(r2) :
        raise ValueError, 'r1 and r2 do not have the same type'

    # if the two objects are dictionaries, then we can handle the case that
    # the lengths are not equal:
    if len(r1) != len(r2) :
        print 'unequal lengths for r1 and r2'
        if type(r1) != type({}) :
            raise ValueError, 'Cannot handle unequal length when it is not a dict'
        keys1 = r1.keys()
        keys2 = r2.keys()
        common = misc.intersect(keys1, keys2)
        r1new = {}
        r2new = {}
        for key in common :
            r1new[key] = r1[key]
            r2new[key] = r2[key]
        r1 = r1new
        r2 = r2new

    if type(r1) == type({}) :
        if r1.keys() != r2.keys() :
            raise ValueError, 'r1 and r2 do not have the same keys'
        I = r1.keys()
    else :
        I = range(len(r1))
    if r1[I[0]].__class__.__name__ == 'Results' or r1[I[0]].__class__.__name__ == 'Container' :
        p1 = misc.extractAttribute(r1, statistic)
        p2 = misc.extractAttribute(r2, statistic)
    else :
        p1 = r1
        p2 = r2

    if type(p1) == type({}) :
        p1 = p1.values()
        p2 = p2.values()

    #import stats
    
    import salstat_stats
    test = salstat_stats.TwoSampleTests(p1, p2)
    test.SignedRanks (p1, p2)

    p = test.prob
    median1 = numpy.median(numpy.array(p1))
    median2 = numpy.median(numpy.array(p2))

    return p, (median1,median2)
    

def confmat(L1, L2) :
    """computes the confusion matrix between two labelings
    """
    
    if len(L1) != len(L2):
        raise ValueError, "labels not the same length"

    n = len(L1)

    classes1 = misc.unique(L1)
    classes2 = misc.unique(L2)
    classes1.sort()
    classes2.sort()
    numClasses1 = len(classes1)
    numClasses2 = len(classes2)

    I1 = {}  # a mapping from classes1 to 0..numclasses1-1
    I2 = {}
    for i in range(numClasses1) :
        I1[classes1[i]] = i
    for i in range(numClasses2) :
        I2[classes2[i]] = i
    
    confmat = numpy.zeros((numClasses1, numClasses2))

    for i in range(n):
        confmat[I1[L1[i]]][I2[L2[i]]] += 1
        
    return confmat


def superConfmat(Y1, Y2, numClasses = 0) :
    """computes the confusion matrix between two labelings, where
    the matrix is assumed to be square, according to the labels of L1
    L1 and L2 are assumed to have integer components in the range
    0,.., numClasses
    """

    if len(Y1) != len(Y2):
        raise ValueError, "labels not the same length"

    n = len(Y1)

    m = max(max(Y1), max(Y2), numClasses)

    # prefer using a list object rather than Numeric object so that there
    # wouldn't be a problem in pickling the object
    confmat = misc.matrix((m, m), 0)

    for i in range(n):
        confmat[Y1[i]][Y2[i]] += 1
        
    return confmat


class ResultsContainer (object) :

    def __len__(self) :

        return len(self.Y)

    def appendPrediction(self, arg, data, pattern) :

        raise NotImplementedError


class Results (list) :

    def __init__(self, arg = None, classifier = None, **args) :
        
        list.__init__(self)

    def __len__(self) :

        return sum([len(res) for res in self])

    def appendPrediction(self, arg, data, pattern) :

        self[-1].appendPrediction(arg, data, pattern)

    def computeStats(self) :

        for results in self :
            results.computeStats()

    def getNumFolds(self) :
        return list.__len__(self)

    def setNumFolds(self) :
        raise AttributeError

    numFolds = property(getNumFolds, setNumFolds, None, 'number of folds')

    def __getattr__(self, attr) :

        if hasattr(self[0], attr) :
            if self.numFolds == 1 :
                return getattr(self[0], attr)
            else :
                if attr not in self.attributeAction :
                    return [getattr(results, attr) for results in self]
                elif self.attributeAction[attr] == 'average' :
                    return numpy.average([getattr(results, attr)
                                          for results in self])
                elif self.attributeAction[attr] == 'returnFirst' :
                    return getattr(self[0], attr)
                elif self.attributeAction[attr] == 'addMatrix' :
                    out = numpy.array(getattr(self[0], attr))
                    for results in self[1:] :
                        out += getattr(results, attr)
                    return out
        else :
            raise AttributeError, 'unknown attribute ' + attr



    def get(self, attribute, fold = None) :

        if fold is None :
            if self.numFolds == 1 :
                return getattr(self[0], attribute)
            else :
                return getattr(self, attribute)
        else :
            return getattr(self[fold], attribute)

    def getDecisionFunction(self, fold = None) :
        return self.get('decisionFunc', fold)

    def getPatternID(self, fold = None) :
        return self.get('patternID', fold)

    def getLog(self, fold = None) :

        if fold is None :
            return [self.get('log', fold_) for fold_ in range(self.numFolds)]
        else :
            return self.get('log', fold)        

    def getInfo(self, fold = None) :

        if fold is None :
            return [self.get('info', fold_) for fold_ in range(self.numFolds)]
        else :
            return self.get('info', fold)


    def convert(self, *options) :

        return [results.convert(*options) for results in self]
    
    def save(self, fileName, *options) :
        """
        save Results to a file
        only attributes given in the attribute list are saved

        OPTIONS::

          'long' - use long attribute list
          'short' - use short attribute list
          using the short attribute list won't allow you to reconstruct
          the Results object afterwards, only the statistics that characterize
          the results.
        """

        resultsList = self.convert(*options)

        myio.save(resultsList, fileName)


class ClassificationFunctions (object) :

    def __repr__(self) :

        if self.numClasses == 1 : return ''

        if not hasattr(self, 'confusionMatrix') :
            try :
                self.computeStats()
            except :
                pass
        if not hasattr(self, 'confusionMatrix') :
            return 'results on unlabeled data, so nothing to show'

        rep = []
        
        rep.extend( self.formatConfusionMatrix() )

        rep.append('success rate: %f' % self.successRate)
        rep.append('balanced success rate: %f' % self.balancedSuccessRate)
        #if self.numClasses == 2 :
        #    rep.append('ppv: %f ' % self.ppv + 'sensitivity: %f' % self.sensitivity)

        if self.numClasses == 2 :
            rep.append('area under ROC curve: %f' % self.roc)
            if int(self.rocN) == self.rocN :
                rep.append('area under ROC %d curve: %f ' % \
                           (self.rocN, getattr(self, 'roc' + str(self.rocN))) )
            else :
                rep.append('area under ROC %f curve: %f ' % \
                           (self.rocN, getattr(self, 'roc' + str(self.rocN))) )

        return '\n'.join(rep)


    def formatConfusionMatrix(self) :

        rep = []
        columnWidth = 4
        for label in self.classLabels :
            if len(label) > columnWidth : columnWidth = len(label)
        columnWidth +=1
        columnWidth = max(columnWidth,
                          math.ceil(math.log10(numpy.max(self.confusionMatrix)) + 1))

        rep.append('Confusion Matrix:')
        rep.append(' ' * columnWidth + ' Given labels:')
        rep.append(' ' * columnWidth + 
		   ''.join([label.center(columnWidth) for label in self.classLabels]))

        (numClassesTest, numClassesTrain) = numpy.shape(self.confusionMatrix)
        for i in range(numClassesTest) :
            label = self.classLabels[i]
            rep.append(label.rjust(columnWidth) + ''.join(
		    [str(self.confusionMatrix[i][j]).center(columnWidth)
		     for j in range(numClassesTrain)]))

        return rep

    def successRates(self) :

        targetClass = 1
        classSuccess = numpy.zeros(self.numClasses, numpy.float_)
        classSize = numpy.zeros(self.numClasses, numpy.int_)
        for i in range(len(self)) :
            classSize[self.givenY[i]] += 1
            if self.givenY[i] == self.Y[i] :
                classSuccess[self.Y[i]] += 1
        balancedSuccess = 0.0
        for i in range(self.numClasses) :
            if classSize[i] > 0 :
                balancedSuccess += classSuccess[i] / float(classSize[i])
        balancedSuccess /= self.numClasses
        sensitivity = 0
        ppv = 0
        if self.numClasses == 2 :
            if classSuccess[targetClass] > 0 :
                sensitivity = float(classSuccess[targetClass]) /\
                              float(classSize[targetClass])
            numTarget = numpy.sum(numpy.equal(self.Y, targetClass))
            if numTarget == 0 :
                ppv = 0
            else :
                ppv = float(classSuccess[targetClass]) / numTarget


        return 1 - len(self.misclassified) / float(len(self.Y)), balancedSuccess,\
               ppv, sensitivity


class ClassificationResultsContainer (ResultsContainer, ClassificationFunctions) :
    """A class for holding the results of testing a classifier
    """

    shortAttrList = ['info', 'log',
                     'successRate', 'balancedSuccessRate',
                     'roc','roc50',
                     'classLabels', 'confusionMatrix',
                     'ppv', 'sensitivity']
    longAttrList = ['info', 'log',
                    'Y', 'L', 'decisionFunc', 'givenY', 'givenL',
                    'classLabels',
                    'patternID', 'numClasses']

    
    def __init__(self, arg, classifier = None, **args) :

        self.rocN = 50
        if 'rocN' in args :
            self.rocN = args['rocN']

        # deal with the roc options and args :
        if 'rocTargetClass' in args :
            self.targetClass = args['rocTargetClass']
            if type(self.targetClass) == type('') :
                # the following is not optimal:
                self.targetClass = arg.labels.classDict[self.targetClass]
        else :
            self.targetClass = 1

        if 'normalization' in args :
            self.rocNormalization = args['normalization']
        else :
            self.rocNormalization = True

        if (arg.__class__.__name__ == 'Container' or
            arg.__class__ == self.__class__) :
            self.copyConstruct(arg, **args)
            return

        data = arg
        self.Y = []
        self.L = []
        self.decisionFunc = []
        self.patternID = []
        self.givenY = []
        self.givenL = []
        
        self.successRate = 0.0
        self.info = 'dataset:\n' + data.__repr__() + \
                    'classifier:\n' + classifier.__repr__()

        if hasattr(classifier, 'labels') :
            self.classLabels = classifier.labels.classLabels
        elif data.labels.L is not None :
            self.classLabels = data.labels.classLabels
        if hasattr(self, 'classLabels') :
            self.numClasses = len(self.classLabels)

    def copyConstruct(self, other, **args) :

        if not hasattr(other, 'decisionFunc') :
            raise AttributeError, 'not a valid results object'

        if 'patterns' in args :
            p = args['patterns']
            idDict = misc.list2dict(other.patternID, range(len(other.patternID)))
            patterns = [idDict[pattern] for pattern in p
                        if pattern in idDict]
        else :
            patterns = range(len(other.Y))

        self.patternID = [other.patternID[p] for p in patterns]
        self.L = [other.L[p] for p in patterns]
        self.Y = [other.Y[p] for p in patterns]
        self.decisionFunc = [other.decisionFunc[p] for p in patterns]
        self.givenY = [other.givenY[p] for p in patterns]
        self.givenL = [other.givenL[p] for p in patterns]        
        self.rocN = 50
        self.classLabels = copy.deepcopy(other.classLabels)
        self.numClasses = len(self.classLabels)
        self.info = other.info
        try :
            self.log = other.log
        except :
            pass
        self.computeStats()
        
    def __getattr__(self, attr) :

        if attr in ['balancedSuccessRate', 'successRate', 'confusionMatrix'] :
            self.computeStats()
            return getattr(self, attr)

        if not attr.find('roc') == 0 :
            raise AttributeError, 'unknown attribute ' + attr

        if attr == 'roc' :
            rocN = None
        elif attr[-1] == '%' :        # roc1%
            rocN = float(attr[3:-1]) / 100.0
        elif float(attr[3:]) >= 1 :   # roc50
            rocN = int(float(attr[3:]))
        else :
            rocN = float(attr[3:])    # roc0.01 (equivalent to roc1%)

        rocValue = self.getROC(rocN)
        # set the value of the roc so that it does not have to be computed
        # next time it is accessed
        # xxx be careful -- this is o.k. as long as rocN hasn't changed.
        # whenver rocN is changed need to reset roc.
        setattr(self, attr, rocValue)

        return rocValue
        

    def getROC(self, rocN = None) :

        rocTP, rocFP, rocValue = roc_module.roc(self.decisionFunc, self.givenY,
                                                rocN, self.rocNormalization)
        return rocValue
            

    def appendPrediction(self, arg, data, pattern) :
        '''
        add the classification results and labels of a data point
        '''

        (y, f) = arg
        if misc.is_nan(f) :
            warnings.warn("decision function value is a nan, prediction ignored",
                          RuntimeWarning)
            return
        self.Y.append(y)
        self.decisionFunc.append(f)
        self.L.append(self.classLabels[y])
        if hasattr(data.labels, 'patternID') and data.labels.patternID is not None :
            self.patternID.append(data.labels.patternID[pattern])
        if hasattr(data.labels, 'Y') and data.labels.Y is not None :
            self.givenY.append(data.labels.Y[pattern])
            self.givenL.append(data.labels.L[pattern])

    def computeStats(self, **args) :

        if len(self.givenY) == 0 : return
        Y = self.givenY
        self.confusionMatrix = superConfmat(self.Y, self.givenY, self.numClasses)

        self.misclassified = [self.patternID[i] for i in range(len(self.patternID))
                              if self.Y[i] != Y[i]]

        self.successRate, self.balancedSuccessRate, self.ppv, \
            self.sensitivity = self.successRates()

    def convert(self, *options) :

        if 'short' in options :
            attributes = self.shortAttrList
        else :
            attributes = self.longAttrList

        return convert(self, attributes)


class ClassificationResults (Results, ClassificationFunctions) :

    # how to construct an attribute from the results of each fold
    # the default action is to make a list
    attributeAction = {'classLabels' : 'returnFirst',
                       'numClasses' : 'returnFirst',
                       'successRate' : 'average',
                       'balancedSuccessRate' : 'average',
                       'ppv' : 'average',
                       'sensitivity' : 'average',
                       'confusionMatrix' : 'addMatrix'}
    
    def __init__(self, arg = None, classifier = None, **args) :

        Results.__init__(self, arg, classifier, **args)
        if arg.__class__ == self.__class__ or type(arg) == type([]) :
            for r in arg :
                self.append(ClassificationResultsContainer(r, **args))
            self.computeStats()
        elif arg is None :
            pass
        else :
            # construct a blank object:
            self.append(ClassificationResultsContainer(arg, classifier, **args))
	    
    def __repr__(self) :

        return ClassificationFunctions.__repr__(self)

    def __getattr__(self, attr) :

        if attr.find('roc') == 0 :
            return numpy.average([getattr(results, attr)
                                  for results in self])
        else :
            return Results.__getattr__(self, attr)


    def plotROC(self, filename=None, fold = None, **args) :

        rocN = None
        if 'rocN' in args :
            rocN = args['rocN']
        if self.numFolds == 1 :
            # if the results are for a single split
            labels = self.getGivenClass()
            dvals = self.getDecisionFunction()
            rocFP, rocTP, area = roc_module.roc(dvals, labels, rocN)
        elif fold is None :
            # get an averaged ROC curve
            labels = self.getGivenClass()
            dvals = self.getDecisionFunction()
            folds = [(dvals[i], labels[i]) for i in range(len(labels))]
            rocFP, rocTP, area = roc_module.roc_VA(folds, rocN)
        else :
            # plot an ROC plot for the given fold
            if fold > self.numFolds :
                raise ValueError, 'foldNum too large'
            labels = self.getGivenClass(fold)
            dvals = self.getDecisionFunction(fold)
            rocFP, rocTP, area = roc_module.roc(dvals, labels, rocN)
        roc_module.plotROC(rocFP, rocTP, filename)

    def toFile(self, fileName, delim = '\t') :
        """
        save results to a (tab) delimited file

        format is:
        patternID, decision function, predicted class, given class, fold

        :Parameters:
          - `fileName` - file name to which to save the results
          - `delim` - delimiter (default: tab)
        """
	
        outfile = open(fileName, 'w')
        for fold in range(self.numFolds) :
            results = self[fold]
            for i in range(len(results)) :
                outfile.write(
                    delim.join([results.patternID[i],
                                str(results.decisionFunc[i]),
                                results.L[i],
                                results.givenL[i],
                                str(fold + 1)]) + '\n')
			  

    def getPredictedLabels(self, fold = None) :
        return self.get('L', fold)

    def getPredictedClass(self, fold = None) :
        return self.get('Y', fold)        

    def getGivenClass(self, fold = None) :
        return self.get('givenY', fold)        
        
    def getGivenLabels(self, fold = None) :
        return self.get('givenL', fold)
        
    def getROC(self, fold = None) :

        return self.get('roc', fold)

    def getROCn(self, rocN = None, fold = None) :

        if rocN is None : rocN = self.rocN
        return self.get('roc' + str(rocN), fold)

    def getSuccessRate(self, fold = None) :

        return self.get('successRate', fold)

    def getBalancedSuccessRate(self, fold = None) :

        return self.get('balancedSuccessRate', fold)

    def getConfusionMatrix(self, fold = None) :

        return self.get('confusionMatrix', fold)

    def getPPV(self, fold = None) :

        return self.get('ppv', fold)

    def getSensitivity(self, fold = None) :

        return self.get('sensitivity', fold)

    def getClassLabels(self) :

        return self.classLabels
    
def convert (object, attributes) :
    obj = misc.Container()
    obj.addAttributes(object, attributes)
    return obj


def saveResultObjects (objects, fileName, *options) :
    """
    save a list or dictionary of Results objects
    it is o.k. if the list or dictionary is itself a list or dictionary of 
    OPTIONS:
    long - save the long attribute list
    """
    
    if type(objects) == type([]) :
	if type(objects[0]) == type([]) :
	    obj = [ [o.convert(*options) for o in resultsList]
		    for resultsList in objects]
	elif type(objects[0]) == type({}) :
	    obj = []
	    for resultsList in objects :
		object = {}
	else :
	    obj = [o.convert(*options) for o in objects]

    elif type(objects) == type({}) :
	obj = {}
	for rkey in objects :
	    if type(objects[rkey]) == type({}) :
		obj[rkey] = {}
		for key in objects[rkey] :
		    obj[rkey][key] = object[rkey][key].convert(*options)
	    elif type(objects[rkey]) == type([]) :
		obj[rkey] = [ object.convert(*options) for object in objects[rkey] ]
	    else :
		obj[rkey] = objects[rkey].convert(*options)
    else :
        raise ValueError, 'expected either a list or dictionary'

    myio.save(obj, fileName)


class ResultsList (list) :

    def __init__(self, resList = None) :
	
        self.rocN = 50
        if resList is None : return
        for results in resList :
            if type(results) == type([]) :
                self.append(ClassificationResults(results))
            else :
                self.append(results)
        self.computeStats()

    def __repr__(self) :

        rep = []
        rep.append('number of Results objects: %d' % len(self))
        rep.append('success rate: %f (%f)' % (self.successRate, numpy.std(self.successRates)) )
        rep.append('balanced success rate: %f (%f)' % 
                   (self.balancedSuccessRate, numpy.std(self.balancedSuccessRates)) )

        rep.append('area under ROC curve: %f (%f)' % (self.roc, numpy.std(self.rocs)) )
        rep.append('area under ROC %d curve: %f (%f)' % \
                   (self.rocN, numpy.average(self.rocNs), numpy.std(self.rocNs)))
        return '\n'.join(rep)

    def save(self, fileName, *options) :

        if 'short' in options :
            attributes = self[0][0].shortAttrList
        else :
            attributes = self[0][0].longAttrList

        resultsList = [[convert(results, attributes) for results in res]
                       for res in self]
        
        myio.save(resultsList, fileName)

    def computeStats(self) :

        self.balancedSuccessRates = [res.balancedSuccessRate for res in self]
        self.balancedSuccessRate = numpy.average(self.balancedSuccessRates)
        self.successRates = [res.successRate for res in self]
        self.successRate = numpy.average(self.successRates)
        self.rocs = [res.roc for res in self]
        self.roc = numpy.average(self.rocs)
        self.rocNs = [res.getROCn(self.rocN) for res in self]


class RegressionResultsContainer (ResultsContainer) :

    def __init__(self, arg, classifier = None, **args) :

        self.Y = []
        self.givenY = []
        self.patternID = []
        self.info = 'dataset:\n' + arg.__repr__() + \
                    'classifier:\n' + classifier.__repr__()


    def appendPrediction(self, y, data, pattern) :

        self.Y.append(y)
        if hasattr(data.labels, 'patternID') and data.labels.patternID is not None :
            self.patternID.append(data.labels.patternID[pattern])
        if hasattr(data.labels, 'Y') and data.labels.Y is not None :
            self.givenY.append(data.labels.Y[pattern])

    
    def computeStats(self) :

        if len(self.givenY) == 0 : return
        self.rmse = math.sqrt(numpy.average([(self.givenY[i] - self.Y[i])**2 
                                             for i in range(len(self.Y))]))


class RegressionResults (Results) :

    attributeAction = {'rmse' : 'average'}

    def __init__(self, arg = None, classifier = None, **args) :

        Results.__init__(self, arg, classifier, **args)
        if arg.__class__ == self.__class__ or type(arg) == type([]) :
            for r in arg :
                self.append(RegressionResultsContainer(r, **args))
            self.computeStats()
        elif arg is None :
            pass
        else :
            # construct a blank object:
            self.append(RegressionResultsContainer(arg, classifier, **args))

    def __repr__(self) :

        rep = []
        rep.append('rmse: ' + str(self.rmse) )

        return ''.join(rep)

    def getRMSE(self, fold = None) :

        return self.get('rmse', fold)

    def getGivenLabels(self, fold = None) :

        return self.get('givenY', fold)
        
    def getDecisionFunction(self, fold = None) :
        
        return self.get('Y', fold)



def loadResults(fileName, isNewFormat = True) :
    """
    isNewFormat -- whether the Results were saved under version 0.6.1 or newer
    """

    res = myio.load(fileName)
    if not isNewFormat :
        return ClassificationResults([res])
    if type(res) == type({}) :
        results = {}
        for key in res :
            results[key] = ClassificationResults(res[key])
        return results
    if type(res[0]) == type([]) :
        return ResultsList(res)
    else :
        return ClassificationResults(res)

def loadResults2(fileName) :
    """
    load a list of list of Results objects or a dictionary of a list of Results objects
    """
    res = myio.load(fileName)
    if type(res) == type({}) :
        results = {}
        for key in res :
            results[key] = [ Results(object) for object in res[key] ]
        return results
    elif type(res) == type([]) :
        return [ [Results(object) for object in listOfResults] for listOfResults in res ]



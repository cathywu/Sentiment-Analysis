import numpy
import random
import math
import os
import tempfile
import copy
import time

from PyML.utils import myio,misc
from PyML.evaluators import resultsObjects

"""functionality for assessing classifier performance"""

__docformat__ = "restructuredtext en"

def test(classifier, data, **args) :
    """test a classifier on a given dataset
    
    :Parameters:
      - `classifier` - a trained classifier
      - `data` - a dataset
      
    :Return:
      a Results class instance

    :Keywords:
      - `stats` - whether to compute the statistics of the match between the
        predicted labels and the given labels [True by default]
    """

    if 'verbose' in args :
        verbose = args['verbose']
    else :
        verbose = 1
        
    if verbose :
        print 'testing', \
              '***********************************************************'

    testStart = time.clock()

    if data.testingFunc is not None :
        data.test(classifier.trainingData, **args)

    classifier.verifyData(data)

    res = classifier.resultsObject(data, classifier, **args)

    for i in range(len(data)) :
        if verbose and i % 100 == 0 and i > 0 :
            print i, 'patterns classified'
        res.appendPrediction(classifier.classify(data, i), data, i)

    try :
        res[0].log = classifier.log
    except :
        pass
    try :
        computeStats = args['stats']
    except :
        computeStats = True
    if computeStats and data.labels.L is not None :
        res.computeStats()

    try :
        res[0].log.testingTime = time.clock() - testStart
    except :
        pass

    return res


def trainTest(classifierTemplate, data, trainingPatterns, testingPatterns, **args) :
    """Train a classifier on the list of training patterns, and test it 
    on the test patterns
    """

    if 'verbose' in args :
        verbose = args['verbose']
    else :
        verbose = True

    trainingData = data.__class__(data, deepcopy = classifierTemplate.deepcopy,
                                  patterns = trainingPatterns)

    classifier = classifierTemplate.__class__(classifierTemplate)

    if verbose :
        print 'training', \
              '***********************************************************'

    classifier.train(trainingData, **args)

    testingData = data.__class__(data, deepcopy = True,
                                 patterns = testingPatterns)

    return classifier.test(testingData, **args)


def loo(classifier, data, **args) :
    """perform Leave One Out 

    :Returns:
      a results object

    USAGE: loo(classifier, data)
    """

    looResults = classifier.resultsObject()
    args['stats'] = False

    for i in range(len(data)) :
        trainingPatterns = misc.setminus(range(len(data)), [i])
        looResults.extend(
            classifier.trainTest(data, trainingPatterns, [i], **args))

    looResults.computeStats()
   
    return looResults


def cvFromFolds(classifier, data, trainingPatterns, testingPatterns,
                **args) :
    
    """perform cross validation

    :Parameters:
      - `classifier` - a classifier template
      - `data` - a dataset
      - `trainingPatterns` - a list providing the training examples for each fold
      - `testingPatterns` - a list providing the testing examples for each fold

    :Keywords:
      - `intermediateFile` - a file name to save intermediate results under
        if this argument is not given, no intermediate results are saved

    :Returns:
      a Results object.
      The ROC curve is computed using the resulting classification of each
      point in the dataset (in contrast to Provost, Fawcett and Kohavi who compute
      average ROC curves).
    """

    assert len(trainingPatterns) == len(testingPatterns)

    cvResults = classifier.resultsObject()
    args['stats'] = False
    
    for fold in range(len(trainingPatterns)) :
        cvResults.extend(trainTest(classifier, data,
                                   trainingPatterns[fold], testingPatterns[fold], **args))
        #if 'intermediateFile' in args :
        #    cvResults.save(args['intermediateFile'])

    cvResults.computeStats()
        
    return cvResults


def cv(classifier, data, numFolds = 5, **args) :
    """perform k-fold cross validation

    :Parameters:
      - `classifier` - a classifier template
      - `data` - a dataset
      - `numFolds` - number of cross validation folds (default = 5)

    :Returns:
      a Results object.

    :Keywords:
      - `numFolds` - number of cross validation folds (default = 5)
      - `seed` - random number generator seed
      - `foldsToPerform` - number of folds to actually perform (in case you're doing
        n fold CV, and want to save time, and only do some of the folds)
    """

    if 'numFolds' in args :
        numFolds = args['numFolds']
    if 'seed' in args :
        random.seed(args['seed'])
    foldsToPerform = numFolds
    if 'foldsToPerform' in args :
        foldsToPerform = args['foldsToPerform']
    if foldsToPerform > numFolds :
        raise ValueError, 'foldsToPerform > numFolds'

    perm = range(len(data))
    random.shuffle(perm)
    foldSize = len(data) / numFolds
    trainingPatterns = []
    testingPatterns = []

    for fold in range(foldsToPerform) :
        if fold < numFolds-1:
            testingPatterns.append(perm[foldSize * fold : foldSize * (fold + 1)])
        else:
            testingPatterns.append(perm[foldSize * fold : len(data)])
        trainingPatterns.append(misc.setminus(range(len(data)),
                                              testingPatterns[-1]))
        
    return cvFromFolds(classifier, data, trainingPatterns, testingPatterns, **args)

        
def stratifiedCV(classifier, data, numFolds = 5, **args) :
    """perform k-fold stratified cross-validation; in each fold the number of
    patterns from each class is proportional to the relative fraction of the
    class in the dataset

    :Parameters:
      - `classifier` - a classifier template
      - `data` - a dataset
      - `numFolds` - number of cross validation folds (default = 5)
      
    :Returns:
      a Results object.

    :Keywords:
      - `numFolds` - number of cross-validation folds -- overrides the numFolds parameter
      - `seed` - random number generator seed
      - `trainingAllFolds` - a list of patterns that are to be used as training
        examples in all CV folds.
      - `intermediateFile` - a file name to save intermediate results under
        if this argument is not given, not intermediate results are saved
      - `foldsToPerform` - number of folds to actually perform (in case you're doing
        n fold CV, and want to save time, and only do some of the folds)
    """

    if 'numFolds' in args :
        numFolds = args['numFolds']
    if 'seed' in args :
        random.seed(args['seed'])
    if 'trainingAllFolds' in args :
        trainingAllFolds = args['trainingAllFolds']
    else :
        trainingAllFolds = []
    foldsToPerform = numFolds
    if 'foldsToPerform' in args :
        foldsToPerform = args['foldsToPerform']
    if foldsToPerform > numFolds :
        raise ValueError, 'foldsToPerform > numFolds'

    trainingAllFoldsDict = misc.list2dict(trainingAllFolds)

    labels = data.labels
    p = [[] for i in range(labels.numClasses)] 
    classFoldSize = [int(labels.classSize[k] / numFolds) for k in range(labels.numClasses)]

    for i in range(len(data)):
        if i not in trainingAllFoldsDict :
            p[labels.Y[i]].append(i)
    for k in range(labels.numClasses):
        random.shuffle(p[k])

    trainingPatterns = [[] for i in range(foldsToPerform)]
    testingPatterns = [[] for i in range(foldsToPerform)]
    for fold in range(foldsToPerform) :
        for k in range(labels.numClasses) :
            classFoldStart = classFoldSize[k] * fold
            if fold < numFolds-1:
                classFoldEnd = classFoldSize[k] * (fold + 1)
            else:
                classFoldEnd = labels.classSize[k]
            testingPatterns[fold].extend(p[k][classFoldStart:classFoldEnd])
            if fold > 0:
                trainingPatterns[fold].extend(p[k][0:classFoldStart] +
                                              p[k][classFoldEnd:labels.classSize[k]])
            else:
                trainingPatterns[fold].extend(p[k][classFoldEnd:labels.classSize[k]])

    if len(trainingPatterns) > 0 :
        for fold in range(len(trainingPatterns)) :
            trainingPatterns[fold].extend(trainingAllFolds)
        
    return cvFromFolds(classifier, data, trainingPatterns, testingPatterns, **args)


def nCV(classifier, data, **args) :
    """
    runs CV n times, returning a 'ResultsList' object.

    :Parameters:
      - `classifier` - classifier template
      - `data` - dataset
      
    :Keywords:
      - `cvType` - which CV function to apply (default: stratifiedCV)
      - `seed` - random number generator seed (default: 1)
        This is used as the seed for the first CV run.  Subsequent runs
        use seed + 1, seed + 2...
      - `iterations` - number of times to run CV (default: 10)
      - `numFolds` - number of folds to use with CV (default: 5)
      - `intermediateFile` - a file name to save intermediate results under
        if this argument is not given, no intermediate results are saved

    :Returns:
      `ResultsList` - a list of the results of each CV run as a ResultsList object
    """

    cvList = resultsObjects.ResultsList()

    cvType = 'stratifiedCV'
    if 'cvType' in args : cvType = args['cvType']
    seed = 1
    if 'seed' in args : seed = args['seed']
    numFolds = 5
    if 'numFolds' in args : numFolds = args['numFolds']
    iterations = 10
    if 'iterations' in args : iterations = args['iterations']
    intermediateFile = None
    if 'intermediateFile' in args : intermediateFile = args['intermediateFile']

    for i in range(iterations) :
        if cvType == 'stratifiedCV' :
            cvList.append(classifier.stratifiedCV(data, numFolds=numFolds, seed=seed + i))
        elif cvType == 'cv' :
            cvList.append(classifier.cv(data, numFolds=numFolds, seed=seed + i))
        else :
            raise ValueError, 'unrecognized type of CV'
        if intermediateFile is not None :
            cvList.save(intermediateFile)

    cvList.computeStats()

    return cvList
    

def makeFolds(data, numFolds, datasetName, directory = '.') :

    '''split a dataset into several folds and save the training and testing
    data of each fold as a separate dataset

    data - a dataset instance
    numfolds - number of folds into which to split the data
    datasetName - string to use for the file names
    directory - the directory into which to deposit the files
    '''
        
    perm = range(len(data))
    random.shuffle(perm)
    foldSize = len(data) / numFolds
    
    for fold in range(numFolds) :
        if fold < numFolds-1:
            testingPatterns = perm[foldSize * fold : foldSize * (fold + 1)]
        else:
            testingPatterns = perm[foldSize * fold : len(data)]
        trainingPatterns = misc.setminus(range(len(data)), testingPatterns)

        trainingData = data.__class__(data, patterns = trainingPatterns)
        testingData = data.__class__(data, patterns = testingPatterns)

        testingDataName = os.path.join(directory, datasetName + 'Testing' + str(fold) + '.data')
        testingData.save(testingDataName)
        trainingDataName = os.path.join(directory, datasetName + 'Training' + str(fold) + '.data')
        trainingData.save(trainingDataName)
    

def cvFromFile(classifier, trainingBase, testingBase, datasetClass, **args) :
    """perform CV when the training and test data are in files whose names
    are of the form:
    trainingBase + number + string
    and
    testingBase + number + string
    For example:
    training0.data, training1.data, training2.data
    and
    testing0.data, testing1.data, testing2.data
    for 3 fold CV.
    training and testing files are matched by the number appearing after
    the strings trainingBase and testingBase
    both trainingBase and testingBase can be paths.
    """

    args['stats'] = False
    import re
    directory = os.path.dirname(trainingBase)
    if directory == '' : directory = '.'
                
    files = os.listdir(directory)
    trainingFiles = [file for file in files
                     if file.find(trainingBase) == 0]
    testingFiles = [file for file in files
                    if file.find(testingBase) == 0]

    # now we check if the training files match the test files:
    numberRE = re.compile(r'\d+')

    trainingNum = [numberRE.findall(trainingFile)[-1]
                   for trainingFile in trainingFiles]
    testingNum = [numberRE.findall(testingFile)[-1]
                  for testingFile in testingFiles]    

    assert len(trainingNum) == len(testingNum)
    for i in range(len(trainingNum)) :
        if trainingNum[i] != testingNum[i] :
            raise ValueError, 'training files do not match testing files'

    trainingData = datasetClass(trainingFiles[0])

    cvResults = classifier.resultsObject(trainingData, classifier)

    for fold in range(len(trainingFiles)) :
        if fold > 0 :
	    trainingData = datasetClass(trainingFiles[fold])

        classifier.train(trainingData)
        del trainingData
        
        testingData = datasetClass(testingFiles[fold])

        r = classifier.test(testingData, **args)
        cvResults.extend(r)

    cvResults.computeStats()
        
    return cvResults



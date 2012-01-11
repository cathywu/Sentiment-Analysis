
import numpy
import os

from PyML.utils import misc
from baseClassifiers import Classifier
from composite import CompositeClassifier
from PyML.evaluators import assess
from PyML.containers.labels import oneAgainstRest

"""classes for multi-class classification"""

__docformat__ = "restructuredtext en"

class OneAgainstOne (CompositeClassifier) :

    '''One-against-one Multi-class classification
    using a two class classifier.
    
    For a k class problem k(k-1) binary classes are trained for all
    pairs of classes; an instance is classified to the class that
    receives the highest number of votes; an instance is constructed
    using a classifier that is used as a template for constructing
    the actual classifiers.
    '''

    def train(self, data, **args) :
        '''train k(k-1)/2 classifiers'''

        Classifier.train(self, data, **args)
        numClasses = self.labels.numClasses

        if numClasses <= 2:
            raise ValueError, 'Not a multi class problem'

        self.classifiers = misc.matrix((numClasses, numClasses))
        for i in range(numClasses - 1) :
            for j in range(i+1, numClasses) :
                self.classifiers[i][j] = self.classifier.__class__(self.classifier)
                dataij=data.__class__(data, deepcopy = self.classifier.deepcopy,
                                      classID = [i,j])
                self.classifiers[i][j].train(dataij)
        self.log.trainingTime = self.getTrainingTime()                

    def classify(self, data, p):
        '''Suppose that x is classified to class c, then the margin is
        defined as the minimum margin found against the k-1 other classes
        '''
        
        numClasses = self.labels.numClasses
        r = numpy.zeros((numClasses, numClasses),numpy.float_)
        vote = numpy.zeros(numClasses)
        for i in range(numClasses - 1) :
            for j in range(i+1, numClasses) :
                r[i][j] = self.classifiers[i][j].decisionFunc(data, p)
                # afterwards I take the minimum or r, so assign:
                r[j][i] = r[i][j]  
                if r[i][j] > 0 :
                    vote[j] += 1
                else:
                    vote[i] += 1
                    
        maxvote = numpy.argmax(vote)
        return maxvote, numpy.min(numpy.absolute(r[maxvote]))

    def preproject(self, data) :

        for i in range(self.labels.numClasses-1):
            for j in range(i+1, self.labels.numClasses):
                self.classifiers[i][j].preproject(data)
        
    test = assess.test

    
class OneAgainstRest (CompositeClassifier) :

    '''A one-against-the-rest multi-class classifier'''
    
    def train(self, data, **args) :
        '''train k classifiers'''
        Classifier.train(self, data, **args)
        numClasses = self.labels.numClasses
        if numClasses <= 2:
            raise ValueError, 'Not a multi class problem'
        self.classifiers = [self.classifier.__class__(self.classifier)
                            for i in range(numClasses)]
        for i in range(numClasses) :
            # make a copy of the data; this is done in case the classifier modifies the data
            datai = data.__class__(data, deepcopy = self.classifier.deepcopy)
            datai =  oneAgainstRest(datai, data.labels.classLabels[i])
            self.classifiers[i].train(datai)

        self.log.trainingTime = self.getTrainingTime()
        
    def classify(self, data, i) :

        r = numpy.zeros(self.labels.numClasses, numpy.float_)
        for j in range(self.labels.numClasses) :
            r[j] = self.classifiers[j].decisionFunc(data, i)

        return numpy.argmax(r), numpy.max(r)

    def preproject(self, data) :

        for i in range(self.labels.numClasses) :
            self.classifiers[i].preproject(data)

    test = assess.test

    def save(self, fileName) :
        """save the trained classifier to a file.
        assumes the classifier is an SVM"""
        file_handle = open(fileName, 'w')
        for classifier in self.classifiers :
            classifier.save(file_handle)

    def load(self, fileName, data) :
        """load a trained classifier from a file.  Also provide the data on which
        the classifier was trained.  It assumes the underlying binary classifier is
        an SVM"""

        from PyML import svm
        Classifier.train(self, data)
        file_handle = open(fileName)
        numClasses = self.labels.numClasses
        self.classifiers = [self.classifier.__class__(self.classifier)
                            for i in range(numClasses)]
        for i in range(numClasses) :
            datai = data.__class__(data, deepcopy = self.classifier.deepcopy)
            datai =  oneAgainstRest(datai, data.labels.classLabels[i])
            self.classifiers[i] = svm.loadSVM(file_handle, datai)
        

def allOneAgainstRest(classifier, data, resultsFile, numFolds = 5, minClassSize = 8) :

    import myio
    labels = data.labels
    
    results = {}
    if os.path.exists(resultsFile) :
	results = assess.loadResults(resultsFile)
    
    for label in labels.classLabels :
        if label in results : continue
        if (minClassSize is not None and
            labels.classSize[labels.classDict[label]] < minClassSize) : continue
        
        myio.log('class: ' + label + '\n')        
        data = oneAgainstRest(data, label)
	try :
	    results[label] = classifier.stratifiedCV(data, numFolds)
	except :
	    results[label] = None
        assess.saveResultObjects(results, resultsFile)
        data.attachLabels(labels)
    

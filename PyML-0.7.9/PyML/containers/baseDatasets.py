import copy
import numpy

from PyML.containers import ker
from PyML.containers import parsers
from PyML.utils import misc
from PyML.containers.labels import Labels

class BaseDataSet (object) :
    """
    A base class for PyML dataset containers

    """
    type = 'dataset'
    isVector = False     # is the dataset Euclidean

    def __init__(self, arg, **args) :

        self.isTrained = False
        self.isTested = False

        # copy construction:
        if arg.__class__ == self.__class__ :
            self.copyConstruct(arg, **args)
            return
        # construct from a file:
        elif type(arg) == type('') :
            self.constructFromFile(arg, **args)
        # construct from a list or numpy array:
        elif (type(arg) == type([]) or type(arg) == numpy.ndarray) :
            self.fromArray(arg, **args)
        # construct a dataset with no features; arg is the number of examples:
        elif type(arg) == int :
            self.makeEmpty(arg, **args)
        else:
            raise ValueError, 'wrong type of argument'
        if 'kernel' in args :
            ker = args['kernel']
            del args['kernel']
            self.attachKernel(ker, **args)
        else :
            self.attachKernel('linear')

    def setTrainingFunc(self, func) :
        #assert func is None or type(func).__name__ == 'function'
        self._trainingFunc = func

    def getTrainingFunc(self) :
        if hasattr(self, '_trainingFunc') :
            return self._trainingFunc
        else :
            return None
                   
    trainingFunc = property(getTrainingFunc, setTrainingFunc,
                            None, '_trainingFunc')

    def setTestingFunc(self, func) :
        #assert func is None or type(func).__name__ == 'function'
        self._testingFunc = func

    def getTestingFunc(self) :
        if hasattr(self, '_testingFunc') :
            return self._testingFunc
        else :
            return None
                   
    testingFunc = property(getTestingFunc, setTestingFunc,
                           None, '_testingFunc')

    def train(self, **args) :

        if self.trainingFunc is not None and not self.isTrained :
            self.trainingFunc(**args)
            self.isTrained = True

    def test(self, trainingData, **args) :

        if self.testingFunc is not None and not self.isTested :
            self.testingFunc(trainingData, **args)
            self.isTested = True
            
    def registerAttribute(self, attributeName, attributeValue = None, action = None) :

        if not hasattr(self, '_registeredAttributes') :
            self._registeredAttributes = [attributeName]
        else :
            self._registeredAttributes.append(attributeName)
        if attributeValue is not None :
            setattr(self, attributeName, attributeValue)
        if not hasattr(self, '_actions') : self._actions = {}
        self._actions[attributeName] = action
            
    def copyConstruct(self, other, **args) :

        forgetClassLabels = False
        if "patterns" in args:
            patterns = args['patterns']
            # if the patterns are ids (strings) convert them to indices:
            if type(patterns[0]) == type('') :
                idDict = misc.list2dict(patterns)
                patternsToCopy = [i for i in range(len(other))
                                  if other.labels.patternID[i] in idDict]
            else :
                patternsToCopy = patterns
        elif "classes" in args :
            patternsToCopy = [i for i in range(len(other))
                              if other.labels.L[i] in args["classes"]]
            forgetClassLabels = True
        elif "classID" in args :
            patternsToCopy = [i for i in range(len(other))
                              if other.labels.Y[i] in args["classID"]]
            forgetClassLabels = True
        else :
            patternsToCopy = range(len(other))

        self.setTrainingFunc(other.trainingFunc)
        self.setTestingFunc(other.testingFunc)

        deepcopy = True
        if 'deepcopy' in args : deepcopy = args['deepcopy']
        # class dependent copying of data:
        self.copy(other, patternsToCopy, deepcopy)

        self.attachKernel(other)
        self.attachLabels(Labels(other.labels,
                                 patterns = patternsToCopy,
                                 forgetClassLabels = forgetClassLabels))

        # copy the registered attribute:
        if hasattr(other, '_registeredAttributes') :
            self._registeredAttributes = other._registeredAttributes[:]
            self._actions = copy.deepcopy(other._actions)
            for attr in self._registeredAttributes :
                a = getattr(other, attr)
                if type(a) == type([]) :
                    if len(a) != len(other) :
                        raise ValueError, 'attribute has bad length'
                    #BaseDataSet.__setattr__(self, attr,
                    #                        [a[i] for i in patternsToCopy])
                    setattr(self, attr, [a[i] for i in patternsToCopy])
                elif hasattr(a, 'type') and a.type == 'dataset' and len(a) == len(self) :
                    acopy = a.__class__(a, patterns = patternsToCopy)
                    setattr(self, attr, acopy)
                else :
                    setattr(self, attr, a)

    def copy(self, other, patterns, deepcopy) :
        """
        Each class that wants to use the generic copy constructor needs
        to define this function for doing class-specific copying"""

        raise NotImplementedError

    def getKernelMatrix(self) :
        """ 
        returns the kernel matrix as a numpy array
        """

        kvec = self.getKernelMatrixAsVector()
        return numpy.reshape(kvec, (len(self), len(self)))

    def attachKernel(self, kernel = 'linear', **args) :

        if type(kernel) == type('') :
            kernel = kernel.lower()
            if kernel == 'linear' or kernel == 'lin' :
                self.kernel = ker.Linear()
            elif kernel == 'polynomial' or kernel == 'poly' :
                self.kernel = ker.Polynomial(**args)
            elif kernel == 'rbf' or kernel == 'gaussian' :
                self.kernel = ker.Gaussian(**args)
            else :
                raise ValueError, 'unrecognized type of kernel'

        elif hasattr(kernel, 'type') and kernel.type == 'dataset' :
            data = kernel
            self.kernel = data.kernel.__class__(data.kernel)
        elif hasattr(kernel, 'type') and kernel.type == 'kernel' :
            self.kernel = kernel.__class__(kernel)
        
    def attachLabels(self, labels) :

        if labels.__class__.__name__ == 'Labels' :
            pass
        elif type(labels) == type('') :
            labels = Labels(labels)
        else :
            raise ValueError, 'wrong type of labels object'
        if len(self) != len(labels) :
            raise ValueError, 'length of labels not equal length of self'
        self.labels = labels

class BaseVectorDataSet (BaseDataSet) :
    """A base class for vector dataset container classes

    Construction::
    
      DataSet(fileName)  -  read data from a file
      DataSet(fileName, classes = listOfClasses) - read only the
      classes that are named in listOfClasses
      DataSet(otherDataSet) - copy construction
      DataSet(otherDataSet, patterns = listOfPatterns) - copy construction
      using a list of patterns to copy
      DataSet(otherDataSet, classes = classesToCopy) - copy construction
      using a list of classes to copy

    Keywords::
    
      deepcopy - whether to deepcopy a dataset (default = True)
      The only container that implements a shallow copy is the SparseDataSet.

    Usage/attributes::
    
      len(dataset) - the number of patterns
      numFeatures - the number of features in the data (when applicable)
    """

    isVector = True     # is the dataset Euclidean
    verbose = 1

    def __init__(self, arg=None, **args):

        self.featureID = None
        BaseDataSet.__init__(self, arg, **args)
        
    def constructFromFile(self, fileName, **args) :

        parser = parsers.parserDispatcher(fileName, **args)
        # the DataSet container can only be used with a csv type file:
        if parser.__class__.__name__ == 'SparseParser' and \
                self.__class__.__name__ == 'DataSet' :
            raise ValueError, \
                'cannot use a DataSet container with a sparse file'
        parser.scan()

        self.initializeDataMatrix(len(parser), len(parser._featureID))

        # read the patterns :
        i = 0
        for x in parser :
            self.addPattern(x, i)
            i += 1
            if i % 100 == 0 :
                print 'read',i,'patterns'

        # postprocessing:
        L = parser._labels
        patternID = parser._patternID
        if patternID is None or len(patternID) == 0 :
            patternID = [str(i) for i in range(1, len(self) + 1)]
        self.featureID, featureKey, featureKeyDict = parser.postProcess()
        if self.__class__.__name__ == 'PySparseDataSet' :
            self.featureKey = featureKey
            self.featureKeyDict = featureKeyDict

        self.updateFeatureDict()

        self.featureIDcompute()
        print 'read', len(self), 'patterns'

        if 'labelsFile' in args :
            self.attachLabels(Labels(args['labelsFile'], **args))
        else :
            self.attachLabels(Labels(L, patternID = patternID, **args))

    def makeEmpty(self, size, **args) :

        L = None
        patternID = None
        if 'labels' in args :
            L = args['labels'].L[:]
            patternID = args['labels'].patternID[:]
        if 'L' in args :
            L = args['L']
        if 'patternID' in args :
            patternID = args['patternID'][:]

        if L is not None : assert size == len(L)                
        if patternID is None :
            patternID = [str(i) for i in range(1, size + 1)]

        self.initializeDataMatrix(size, 0)

        if 'labelsFile' in args :
            self.attachLabels(Labels(args['labelsFile'], **args))
        else :
            args['patternID'] = patternID
            self.attachLabels(Labels(L, **args))
        

    def fromArray(self, X, **args) :

        L = None
        patternID = None
        self.featureID = None
        if 'labels' in args :
            L = args['labels'].L[:]
            patternID = args['labels'].patternID[:]
        if 'L' in args :
            L = args['L']
        if 'patternID' in args :
            patternID = args['patternID'][:]
        if 'featureID' in args :
            if self.__class__.__name__ == 'SparseDataSet' :
                raise vluaeError, 'cannot set feature ID for SparseDataSet'
            self.featureID = args['featureID'][:]

        if L is not None : assert len(X) == len(L)                
        if self.featureID is None :
            self.featureID = [str(i) for i in range(len(X[0]))]
        if patternID is None :
            patternID = [str(i) for i in range(1, len(X) + 1)]

        self.fromArrayAdd(X)
        self.updateFeatureDict()
        self.featureIDcompute()

        if 'labelsFile' in args :
            self.attachLabels(Labels(args['labelsFile'], **args))
        else :
            args['patternID'] = patternID
            self.attachLabels(Labels(L, **args))

    def fromArrayAdd(self, X) :

        if type(X[0]) == dict :
            featureHashDict = {}
            for i in range(len(X)) :
                for key in X[i] :
                    if hash(key) in featureHashDict :
                        if featureHashDict[hash(key)] != key :
                            raise valueError, 'hash clash'
                    else :
                        featureHashDict[hash(key)] = key
            featureHashes = featureHashDict.keys()
            featureHashes.sort()
            self.featureID = [featureHashDict[key] for key in featureHashes]
            self.initializeDataMatrix(len(X), len(self.featureID))
            for i in range(len(X)) :
                x = {}
                for key in X[i] :
                    x[hash(key)] = X[i][key]
                self.addPattern(x, i)
        else :
            self.initializeDataMatrix(len(X), len(X[0]))
            for i in range(len(X)) :
                self.addPattern(X[i], i)
            
    def __repr__(self) :
        
        rep = '<' + self.__class__.__name__ + ' instance>\n'
        rep += 'number of patterns: ' + str(len(self)) +'\n'
        if self.X is not None :
            rep += 'number of features: ' + str(self.numFeatures) + '\n'
        rep += self.labels.__repr__()
        
        return rep

    def save(self, fileName, **args) :
        """save a dataset to a file (does not use pickle!)

        :Parameters:
          - `fileName` - a file name or a file handle

        :Keywords:
          - `format` - 'csv' or 'sparse'; by default format is chosen by the
            type of the dataset -- sparse containers save in sparse format
            and non-sparse containers in csv format.
          - `delimiter` - which delimiter to use when saving in csv format
          - `patterns` - save only those patterns whose indices are given
          - `ids` - save only those patterns whose pattern ID are given
          - `sortByID` - whether to sort the lines according to the pattern ID
            (default = False)
          - `sortByLabel` - whether to sort the lines according to the class label
            (default = False)
        """

        print 'saving to ', fileName
        if type(fileName) == type('') :
            fileHandle = open(fileName, 'w')
        else :
            fileHandle = fileName

        L = self.labels.L

        if self.__class__.__name__.lower().find('sparse') >= 0 :
            format = 'sparse'
        else :
            format = 'csv'
        print 'detected file format as:', format
        if 'format' in args :
            format = args['format']
        if 'delimiter' in args :
            delim = args['delimiter']
        else :
            delim = ','
        if 'patterns' in args :
            patterns = args['patterns']
        else :
            patterns = range(len(self))
        if 'ids' in args :
            idDict = misc.list2dict(args['ids'])
            patterns = [i for i in range(len(self))
                        if self.labels.patternID[i] in idDict]
        if 'sortByID' in args and args['sortByID'] :
            ids = self.labels.patternID[:]
            ids.sort()
            idMap = misc.list2dict(self.labels.patternID, range(len(self)))
            idDict = misc.list2dict(patterns)
            patterns = [idMap[id] for id in ids
                        if idMap[id] in idDict]
        if 'sortByLabel' in args and args['sortByLabel'] :
            y = self.labels.Y[:]
	    patterns = numpy.argsort(self.labels.Y)

        if format == 'csv' :
            if L is None :
                labels = ''
            else :
                labels = 'labels' + delim
            fileHandle.write('#' + 'patternID' + delim + labels + 
                             delim.join(self.featureID) + '\n')
        for i in patterns :
            x = self.getPattern(i)
            if format == 'sparse' :
                if self.labels.patternID is not None :
                    fileHandle.write(str(self.labels.patternID[i]) + ',')
                if L is not None :
                    if type(L[i]) == type([]) :
                        fileHandle.write(';'.join(L[i]) + ' ')
                    else :
                        fileHandle.write(str(L[i]) + ' ')
                if type(x) == type({}) :
                    tokens = [self.featureID[self.featureKeyDict[key]]+':'+
                              str(x[key]) for key in x]
                else :
                    tokens = [self.featureID[i] + ':' + str(x[i])
                              for i in range(self.numFeatures)
                              if x[i] != 0]
                fileHandle.write(' '.join(tokens) + '\n')
            else :
                if self.labels.patternID is not None :
                    fileHandle.write(str(self.labels.patternID[i]) + delim)
                if L is not None :
                    if type(L[i]) == type([]) :
                        fileHandle.write(';'.join(L[i]) + delim)
                    else :
                        fileHandle.write(L[i] + delim)
                if type(x) == type({}) :
                    tokens = [str(x.get(self.featureKey[i],0))
                              for i in range(self.numFeatures)]
                else :
                    tokens = [str(val) for val in x]
                fileHandle.write(delim.join(tokens) + '\n')
        fileHandle.close()

    def getMatrix(self) :

        X = numpy.zeros((len(self), self.numFeatures), float)
        for i in range(len(self)) :
            X[i] = self.getPattern(i)
        return X
            
    def extend(self, other, patterns = None) :

        if self.__class__ != other.__class__ :
            raise ValueError, 'datasets should be the same class'

        if patterns is None : patterns = range(len(other))

        # first check if ids have compatible hash values :
        for id in other.featureID :
            if (hash(id) in self.featureKeyDict and
                id != self.featureID[self.featureKeyDict[hash(id)]]) :
                raise ValueError, 'bad hash'
        
        # recompute featureID related stuff
        self.featureKey = misc.union(self.featureKey, other.featureKey)
        self.featureKey.sort()
        self.featureKeyDict.clear()
        for i in range(len(self.featureKey)) :
            self.featureKeyDict[self.featureKey[i]] = i
        featureIDs = misc.union(self.featureID, other.featureID)
        self.featureID = [None for i in range(len(self.featureKey))]
        for id in featureIDs :
            self.featureID[self.featureKeyDict[hash(id)]] = id

        self.extendX(other, patterns) 
        self.labels.extend(other.labels, patterns)
        #self.attachLabels(self.labels.L)

    def keepFeatures(self, features) :
        """eliminate all but the give list of features
        INPUT:
        features - a list of features to eliminate; these are either numbers
        between 0 and numFeatures-1 (indices of features, not their IDs) or
        featureIDs
        """

        if type(features[0]) == type('') :
            features = self.featureNames2IDs(features)
        self.eliminateFeatures(misc.setminus(range(self.numFeatures), features))

        
    def featureNames2IDs(self, featureList) :
        """convert a list of feature Names into their numeric IDs"""

        return [self.featureDict[feature] for feature in featureList]

    def addFeatureKeyDict(self) :

        self.featureKeyDict = {}
        for i in range(len(self.featureID)) :
            self.featureKeyDict[hash(self.featureID[i])] = i

class WrapperDataSet (BaseDataSet) :

    isWrapper = True

    def __len__(self) :
        """the number of patterns in the dataset"""

        return self.size()

    def getX(self) :

        return None

    def setX(self, value) :

        raise ValueError, 'X cannot be set'

    X = property (getX, setX, None, 'X')

    def get_kernel(self) :

        return self._kernel

    def set_kernel(self, value) :

        raise ValueError, 'kernel cannot be set'

    kernel = property (get_kernel, set_kernel, None, 'kernel')


    def attachLabels(self, labels) :

        BaseDataSet.attachLabels(self, labels)
        if hasattr(self.labels, 'Y') and self.labels.Y is not None :
            for i in range(len(labels)) :
                self.setY(i, labels.Y[i])

    attachKernel = ker.attachKernel


import numpy
from PyML.containers.ext import caggregate
from PyML.containers.baseDatasets import BaseDataSet, WrapperDataSet

class Aggregate (WrapperDataSet, caggregate.Aggregate) :
    """
    combines several C++ dataset objects into a single dataset.
    its dot product is a weighted sum of the kernels of the individual
    dataset objects

    Construction of an aggregate requires a list of dataset objects.
    It is assumed that all datasets refer to the same underlying objects so
    in particular have the same labels and same number of patterns (the labels
    object is initialized using the labels of the first dataset in the list).
    """

    isVector = False
    
    def __init__(self, arg, **args) :
	"""
        :Parameters:
	  - `arg` - either an Aggregate object (for copy construction) or a list
	    of C++ dataset objects

	:Keywords:
	  - `weights` - a list of weights used for computing the dot product
	    element i is the weight for dataset i in the aggregate
	"""
        BaseDataSet.__init__(self, arg, **args)
        self._trainingFunc = self.aggregate_train
        self._testingFunc = self.aggregate_test

    def fromFile(self, file_name) :
        raise NotImplemented, "construction from file not supported for this container"
    
    def fromArray(self, datas, **args) :
        
        self.checkDatas(datas)
        self.pydatas = datas
        if 'weights' in args :
            self.pyweights = args['weights']
            for i in range(len(self.pyweights)) :
                self.pyweights[i] = float(self.pyweights[i])
            assert len(self.pyweights) == len(self.pydatas)
        else :
            self.pyweights = [1.0 / len(self.pydatas) for i in range(len(self.pydatas))]
        caggregate.Aggregate.__init__(self, len(self.pydatas[0]), tuple(self.pyweights))
        #self._addDatas()
        WrapperDataSet.attachLabels(self, self.pydatas[0].labels)
        self.attachKernel('linear')
        
    def _addDatas(self) :
        #caggregate.Aggregate.__init__(self, len(self.pydatas[0]), tuple(self.pyweights))
        for data in self.pydatas :
            self.addDataSet(data.castToBase())

    def addData(self, data, weight) :

        self.pydatas.append(data)
        self.weights.append(weight)
        self.addDataSet(data.castToBase(), float(weight))

    def copy(self, other, patterns, deepcopy) :

        self.pyweights = other.pyweights[:]
        self.pydatas = [data.__class__(data, patterns = patterns)
                        for data in other.pydatas]
        caggregate.Aggregate.__init__(self, len(self.pydatas[0]), tuple(self.pyweights))
        #self._addDatas()

    def __len__(self) :

        return self.size()

    def checkDatas(self, datas) :

        lengths = [len(data) for data in datas]
        if not numpy.alltrue(numpy.equal(lengths, lengths[0])) :
            raise ValueError, 'datasets not equal lengths'
        for i in range(1, len(datas)) :
            if datas[i].labels.patternID != datas[0].labels.patternID :
                raise ValueError, 'datasets not have the same pattern IDs'

    def aggregate_train(self, **args) :
        self._addDatas()        

    def aggregate_test(self, trainingData, **args) :
        self._addDatas()
        

class DataAggregate (BaseDataSet) :

    """An aggregate of datasets.
    a DataAggregate object contains a list of datasets in its datas attribute,
    and behaves like a dataset when it comes to copy construction, so it can
    be used as a dataset object when it comes to testing classifiers.
    USAGE:
    DataAggregate(list) - construct an object out of a list of datasets
    (they do not have to be of the same kind!
    It is assumed that all datasets are the same length, and have the same labels
    DataAggregate(other[,optional arguments]) - copy construction - all options
    supported by the dataset classes can be used.
    """

    def __init__(self, arg, *opt, **args) :

        BaseDataSet.__init__(self)
        if arg.__class__ == self.__class__ :
            other = arg
            self.datas = [other.datas[i].__class__(other.datas[i], *opt, **args)
                          for i in range(len(other.datas))]
        elif type(arg) == type([]) :
            self.datas = arg
        else :
            raise ValueError, 'wrong type of input for DataAggregate'
        self.labels = self.datas[0].labels

    def __len__(self) :

        return len(self.datas[0])

    def __repr__(self) :

        rep = ''
        for i in range(len(self.datas)) :
            rep += str(self.datas[i]) + '\n'

        return rep


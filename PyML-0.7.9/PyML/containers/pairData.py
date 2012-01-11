
from PyML.utils import misc
from PyML.utils import arrayWrap
from PyML.containers.ext import cpairdataset
from PyML.containers.baseDatasets import WrapperDataSet, BaseDataSet
from PyML.containers.labels import Labels

"""
classes for dealing with data that is composed of pairs of simpler objects
for which a kernel is available
"""

__docformat__ = "restructuredtext en"

class PairDataSet (WrapperDataSet, cpairdataset.PairDataSet) :

    """
    DataSet container for pairs of objects.

    The kernel between a pair is defined via the kernel between the
    members of the pair:
    K((X_1,X_2), (X'_1, X'_2)) = K'(X_1, X'_1) K'(X_2, X'_2) +
                                 K'(X_1, X'_2) K'(X_2, X'_1)

    file format::
    
      id1_id2 label,... (can have additional fields that are ignored)    
    
    """

    isVector = False
    
    def __init__(self, arg, **args) :
        """
        :Parameters:
          - `arg` - a file name or another PairDataSet object.
            if a file name is supplied the constructor expects a dataset
            object as a keyword argument 'data'
        :Keywords:
          - `data` - a dataset object from which the kernel between the pairs
            of patterns is derived.
          - `patterns` - patterns to copy when performing copy construction
        """

        BaseDataSet.__init__(self)
        if arg.__class__ == self.__class__ :
            self.copyConstruct(arg, **args)
        elif type(arg) == type('') :
            if 'data' not in args :
                raise ValueError, 'missing data object'
            self._data = args['data']
            self.constructFromFile(arg)
            
        self.attachKernel('linear')

    def copy(self, other, patterns, deepcopy) :

        self.callCopyConstructor(other, patterns)
        self.pairs = [other.pairs[p] for p in patterns]
        self._data = other._data

    def constructFromFile(self, fileName) :

        patternIDdict = misc.list2dict(self._data.labels.patternID,
                                       range(len(self._data)))

        labels = Labels(fileName)
        patterns = []
        pairs = []
        for i in range(len(labels)) :
            p1,p2 = labels.patternID[i].split('_')
            # add only pairs for which we have kernel data:
            if p1 in patternIDdict and p2 in patternIDdict :
                pairs.append((patternIDdict[p1],patternIDdict[p2]))
                patterns.append(i)
            else :
                print p1, ' or ', p2, 'not found'
        labels = labels.__class__(labels, patterns = patterns)

        self.pairs = pairs

        first = [pair[0] for pair in pairs]
        second = [pair[1] for pair in pairs]
        firstVector = arrayWrap.intVector([pair[0] for pair in pairs])
        secondVector = arrayWrap.intVector([pair[1] for pair in pairs])            
        self.callConstructor(firstVector, secondVector)

        WrapperDataSet.attachLabels(self, labels)


    def callConstructor(self, firstVector, secondVector) :
        
        cpairdataset.PairDataSet.__init__(self, firstVector, secondVector,
                                          self._data.castToBase())

    def callCopyConstructor(self, other, patterns) :
        
        cpairdataset.PairDataSet.__init__(self, other, patterns)

    def __len__(self) :

        return len(self.pairs)

    def getPair(self, i) :

        return tuple(self.labels.patternID[i].split())
        
class SimplePairDataSet (BaseDataSet) :

    """
    DataSet container for pairs of objects.

    file format::
    
      id1_id2, label,... (can have additional fields that are ignored)    
    
    """

    isVector = False
    
    def __init__(self, arg, **args) :
        """
        :Parameters:
          - `arg` - a file name or another PairDataSet object.
            if a file name is supplied the constructor expects a dataset
            object as a keyword argument 'data'
        :Keywords:
          - `data` - a dataset object from which the kernel between the pairs
            of patterns is derived.
          - `patterns` - patterns to copy when performing copy construction
        """

        BaseDataSet.__init__(self)
        if arg.__class__ == self.__class__ :
            if 'patterns' in args :
                patterns = args['patterns']
            else :
                patterns = range(len(arg))
            self.copyConstruct(arg, patterns)
        elif type(arg) == type('') :
            if 'data' not in args :
                raise ValueError, 'missing data object'
            self.data = args['data']
            self.constructFromFile(arg)

    def copyConstruct(self, other, patterns) :

        self.pairs = [other.pairs[p] for p in patterns]
        self.data = other.data
        self.labels = Labels(other.labels, patterns = patterns)

    def constructFromFile(self, fileName) :

        delim = ','
        if self.data is not None :
            patternIDdict = misc.list2dict(self.data.labels.patternID,
                                           range(len(self.data)))
        else :
            patternIDdict = {}
            
        L = []
        patternID = []
        pairs = []
        file = open(fileName)
        for line in file :
            tokens = line[:-1].split(delim)
            #patternID.append(tokens[0])
            p1,p2 = tokens[0].split('_')
            if p1 > p2 : p1,p2 = p2,p1
            # add only pairs for which we have kernel data:
            if p1 in patternIDdict and p2 in patternIDdict or self.data is None :
                pairs.append((p1,p2))
                L.append(tokens[1])
                patternID.append('_'.join([p1,p2]))
            else :
                print p1, ' or ', p2, 'not found'
        self.pairs = pairs
        self.labels = Labels(L, patternID = patternID)

    def __len__(self) :

        return len(self.pairs)

    def getPair(self, i) :

        return tuple(self.labels.patternID[i].split())
        
    
class PairDataSetSum (PairDataSet, cpairdataset.PairDataSetSum) :

    def __init__(self, arg, **args) :

        PairDataSet.__init__(self, arg, **args)

    def callConstructor(self, firstVector, secondVector) :
        
        cpairdataset.PairDataSetSum.__init__(self, firstVector, secondVector,
                                             self._data.castToBase())

    def callCopyConstructor(self, other, patterns) :
        
        cpairdataset.PairDataSetSum.__init__(self, other, patterns)

    def dotProduct(self, i, j, other = None) :

        return cpairdataset.PairDataSetSum.dotProduct(self, i, j, other)
    

class PairDataSetOrd (PairDataSet, cpairdataset.PairDataSetOrd) :

    def __init__(self, arg, **args) :

        PairDataSet.__init__(self, arg, **args)

    def callConstructor(self, firstVector, secondVector) :
        
        cpairdataset.PairDataSetOrd.__init__(self, firstVector, secondVector,
                                             self._data.castToBase())

    def callCopyConstructor(self, other, patterns) :
        
        cpairdataset.PairDataSetOrd.__init__(self, other, patterns)

    def dotProduct(self, i, j, other = None) :

        return cpairdataset.PairDataSetOrd.dotProduct(self, i, j, other)
    

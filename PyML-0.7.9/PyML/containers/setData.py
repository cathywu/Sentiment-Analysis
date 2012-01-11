
from PyML.utils import misc
from PyML.utils import arrayWrap
from PyML.containers.ext import csetdata
from PyML.containers.baseDatasets import WrapperDataSet, BaseDataSet
from PyML.containers.labels import Labels

"""
classes for dealing with data that is composed of pairs of simpler objects
for which a kernel is available
"""

__docformat__ = "restructuredtext en"

class SetData (WrapperDataSet, csetdata.SetData) :

    """
    DataSet container for sets of objects.

    The kernel between two sets is defined via the kernel between the
    members of the sets:
    K(s, s') = sum_{x in s, x' in s'} K'(x, x') \ (|s| |s'|)

    file format::
    
      id1 id2 ...idk label
    
    """

    isVector = False
    
    def __init__(self, arg, **args) :
        """
        :Parameters:
          - `arg` - a file name or another SetData object.
            if a file name is supplied the constructor expects a dataset
            object as a keyword argument 'data'
        :Keywords:
          - `data` - a dataset object from which the kernel between the pairs
            of patterns is derived.
          - `patterns` - patterns to copy when performing copy construction
        """

        BaseDataSet.__init__(self, arg, **args)

    def copy(self, other, patterns, deepcopy) :

        self.callCopyConstructor(other, patterns)
        self.n = len(patterns)
        self._data = other._data

    def constructFromFile(self, file_name, **args) :

        if 'data' not in args :
            raise ValueError, 'missing data object'
        self._data = args['data']

        id_dict = misc.list2dict(self._data.labels.patternID,
                                 range(len(self._data)))
        file_handle = open(file_name)
        L = []
        sets = []
        for line in file_handle :
            tokens = line.split()
            sets.append([id_dict[token] for token in tokens[:-1] ])
            L.append(tokens[-1])
        self.n = len(sets)
        self.callConstructor(len(sets))
        for s in sets :
            self.add(tuple(s))
        labels = Labels(L)
        WrapperDataSet.attachLabels(self, labels)

    def callConstructor(self, size) :
        
        csetdata.SetData.__init__(self, size, self._data.castToBase())

    def callCopyConstructor(self, other, patterns) :
        
        csetdata.SetData.__init__(self, other, patterns)

    def __len__(self) :

        return self.n


from PyML.containers.ext import ckerneldata
from PyML.containers.baseDatasets import WrapperDataSet
from PyML.containers.ext import ckernel
from PyML.containers import labels
from PyML.utils import misc,myio
from PyML.utils import arrayWrap

class KernelData (WrapperDataSet, ckerneldata.KernelData) :
    """
    A container for holding a dataset with a dot product derived from
    a pre-computed kernel matrix

    File format:
    delimited file with the first column interpreted as pattern IDs if
    it is non-numeric; comments can appear with # or %
    gist format is accepted as well.
    
    Construction::

      Copy construction:
      KernelData(other) optional keyword arguments are the same as
      other dataset containers

      Construction from file:
      KernelData(matrixFile [,labelsFile = labelsFileName, gistFormat = True])
      matrixFile -- a file with the kernel matrix
      labelsFile -- keyword argument containing a file name with the labels.
      the parser tries to automatically guess if the file is in GIST format;
      in case this is not detected, use the 'gistFormat' keyword argument.
      A matrix file with labels in it is not supported yet.
      additional keyword arguments are the same as those supporting reading
      of delimited files.
    """

    # note the order of the class derivation -- the order matters if an attribute
    # might be defined in more than one of the superclasses

    isVector = False
    
    def __init__(self, arg = None, **args) :

        if arg.__class__ == self.__class__ :
            self.copyConstruct(arg, **args)			
        elif type(arg) == type('') :
            self.constructFromFile(arg, **args)
            WrapperDataSet.attachKernel(self)

    def copy(self, other, patterns, deepcopy) :
        
        ckerneldata.KernelData.__init__(self, other, patterns)		

    def constructFromFile(self, fileName, **args) :
		
        matrix = ckernel.KernelMatrix()
        matrix.thisown = 0
        patternID = []
        delim = None
        delim = misc.getDelim(fileName)
        idColumn = 0
        if 'idColumn' in args :
            idColumn = args['idColumn']
        if idColumn is None :
            firstColumn = 0
        else :
            firstColumn = 1
        print firstColumn
        print idColumn
        matrixFile = myio.myopen(fileName)
        firstRow = True
        for line in matrixFile :
            # skip comments:
            if line[0] in ["%", "#"] : continue
            tokens = line.split(delim)
            # check if the file is in gist format:
            if firstRow :
                firstRow = False
                try :
                    float(tokens[-1])
                except :
                    continue
                if ( ('headerRow' in args and args['headerRow']) or
                     ('gistFormat' in args and args['gistFormat']) ):
                    continue
            values = arrayWrap.floatVector([float(token) for token in tokens[firstColumn:]])
            matrix.addRow(values)
            if idColumn is not None :
                patternID.append(tokens[0])

        ckerneldata.KernelData.__init__(self, matrix)
        if 'labelsFile' in args :
            self.attachLabels(labels.Labels(args['labelsFile'], **args))
        else :
            self.attachLabels(labels.Labels(None, patternID = patternID))



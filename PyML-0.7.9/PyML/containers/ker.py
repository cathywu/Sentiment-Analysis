
import sys
import math
import os

from ext import ckernel
from PyML.utils import misc
from PyML.base.pymlObject import PyMLobject
from ext.ckernel import NONE, COSINE, TANIMOTO, DICES

normalizationMethods = ['none', 'cosine', 'tanimoto', 'dices']

__docformat__ = "restructuredtext en"

"""functionality for dealing with kernels and kernel objects"""

class Kernel (object) :
    """base class for kernel objects

    each kernel class defines an ``eval`` function:
    eval(self, datai, i, j, dataj = None) that evaluates the kernel between
    patterns i and j of dataset ``datai``; if dataj is given then pattern j
    is assumed to come from dataset ``dataj``

    """
    
    type = 'kernel'

    def __repr__(self) :

        rep = '<' + self.__class__.__name__ + ' instance>\n'
        
        return rep

    def dump(self) :
        """
        returns a string that can be used to construct an equivalent object
        """
        kstr = self.__module__ + '.' + self.__class__.__name__ + '(' + \
               self.constructionParams() + ')'

        return kstr

    def constructionParams(self) :

        raise NotImplementedError        

    def eval(self, datai, i, j, dataj = None) :

        """evaluate the kernel function between
        patterns i and j of dataset ``datai``; if dataj is given then pattern j
        is assumed to come from dataset ``dataj``
        """

        raise NotImplementedError


class Linear (Kernel, ckernel.Linear) :
    """A Linear kernel (dot product)

    Construction:
    k = Linear()
    """

    def __init__(self, arg = None, **args) :

        if arg.__class__ == self.__class__ :
            ckernel.Linear.__init__(self, arg)
        else :
            ckernel.Linear.__init__(self)
            if 'normalization' in args :
                self.normalization = args['normalization']
            else :
                self.normalization = NONE

    def constructionParams(self) :

        return ""
    
    def eval (self, datai, i, j, dataj = None) :

        if dataj is None : dataj = datai
        return ckernel.Linear.eval(self,
                                   datai.castToBase(), i, j, dataj.castToBase())

class Cosine (Kernel, ckernel.Cosine) :
    """A Cosine kernel (dot product)
    Construction:
    k = Cosine()
    """
    
    def __init__(self, arg = None) :

        if arg.__class__ == self.__class__ :
            ckernel.Cosine.__init__(self, arg)
        else :
            ckernel.Cosine.__init__(self)        
    
    def constructionParams(self) :

        return ''
    
    def eval (self, datai, i, j, dataj = None) :

        if dataj is None : dataj = datai
        return ckernel.Cosine.eval(self,
                                   datai.castToBase(), i, j, dataj.castToBase())


class Polynomial (Kernel, ckernel.Polynomial) :
    """
    A Polynomial kernel
    K(x,y) = (x \dot y + additiveConst) ** degree

    Construction:
    k = Polynomial(degree, additiveConst)

    Attributes:
    additiveConst, degree - kernel parameters
    """

    attributes = {'normalization' : NONE,
                  'degree' : 2,
                  'additiveConst' : 1.0}
    
    def __init__(self, arg = 2, **args):

        if arg.__class__ == self.__class__ :
            ckernel.Polynomial.__init__(self, arg)
        else :
            ckernel.Polynomial.__init__(self)       
            for attribute in self.attributes :
                if attribute in args :
                    setattr(self, attribute, args[attribute])
                else :
                    setattr(self, attribute, self.attributes[attribute])            
            if arg != 2 :
                self.degree = arg

    def __repr__(self) :

        rep = '<' + self.__class__.__name__ + ' instance>\n'
        rep += 'degree : ' + str(self.degree) + '\n'
        rep += 'affine coefficient : ' + str(self.additiveConst)
        
        return rep

    def constructionParams(self) :

        return 'degree = ' + str(self.degree) + ',' + \
               'additiveConst = ' + str(self.additiveConst)

    def eval (self, datai, i, j, dataj = None) :

        if dataj is None : dataj = datai
        return ckernel.Polynomial.eval(self,
                                       datai.castToBase(), i, j, dataj.castToBase())

class Gaussian (Kernel, ckernel.Gaussian) :

    """
    A Gaussian (RBF) kernel
    K(x,y) = exp( - gamma * ||x - y||**2

    Construction:
    k = Gaussian(gamma)

    Attributes:
    gamma - kernel width parameter
    """

    attributes = {'normalization' : NONE,
                  'gamma' : 1.0,}

    def __init__(self, arg = 1.0, **args) :

        if arg.__class__ == self.__class__ :
            ckernel.Gaussian.__init__(self, arg)
        else :
            ckernel.Gaussian.__init__(self)       
            for attribute in self.attributes :
                if attribute in args :
                    setattr(self, attribute, args[attribute])
                else :
                    setattr(self, attribute, self.attributes[attribute])            
            if arg != 1.0 :
                self.gamma = arg

    def __repr__(self) :
        
        rep = '<' + self.__class__.__name__ + ' instance>\n'
        rep += 'gamma : ' + str(self.gamma)
        
        return rep

    def constructionParams(self) :

        return 'gamma = ' + str(self.gamma)

    def eval (self, datai, i, j, dataj = None) :

        if dataj is None : dataj = datai
        return ckernel.Gaussian.eval(self,
                                     datai.castToBase(), i, j, dataj.castToBase())


def attachKernel(data, kernel = 'linear', **args) :

    if 'normalization' in args :
        if args['normalization'].lower() not in normalizationMethods :
            raise ValueError, 'unrecognized normalization method'
        args['normalization'] = normalizationMethods.index(args['normalization'].lower())
    if type(kernel) == type('') :
        kernel = kernel.lower()
        if kernel == 'linear' :
            k = Linear(**args)
        elif kernel == 'cosine' :
            k = Cosine()
        elif kernel == 'polynomial' or kernel == 'poly' :
            k = Polynomial(**args)
        elif kernel == 'rbf' or kernel == 'gaussian' :
            k = Gaussian(**args)
        else :
            raise ValueError, 'unrecognized type of kernel'
    elif hasattr(kernel, 'type') and kernel.type == 'dataset' :
        other = kernel
        k = other._kernel.__class__(other._kernel)
    elif hasattr(kernel, 'type') and kernel.type == 'kernel' :
        k = eval(kernel.__class__.__name__ + '(kernel)')

    # destroy the data object's kernel if it has one:
    if hasattr(data, '_kernel') :
        data._kernel.thisown = True
        #    del data._kernel
    data._kernel = k
    data._kernel.thisown = False
    data.setKernel(k.castToBase())
    kernelName = k.__class__.__name__
    if kernelName == 'Cosine' or kernelName == 'Gaussian' or k.normalization != NONE :
        data.computeNorms()


def kernel2file(data, fileName, **args) :

    """compute a kernel matrix and save it to a file in tab delimited format

    :Parameters:
      - `data` - a dataset
      - `fileName` - file name to save the kernel

    :Keywords:
      - `format` - the format in which to save the kernel: pyml or gist formats [default: 'gist']
        gist format has an additional header line that contains the ids.
    """

    if fileName is None or fileName == '-' :
        outfile = sys.stdout
        fileName = 'stdout'
    else :
        outfile = open(fileName, 'w')

    format = 'gist'
    if 'format' in args :
        format = args['format']
    import tempfile
    tmpfile = tempfile.mktemp()
    ckernel.kernel2file(data.castToBase(), tmpfile)
    tmp = open(tmpfile)
    outfile = open(fileName, 'w')
    if format == 'gist' :
        outfile.write(fileName + '\t')
        outfile.write('\t'.join(data.labels.patternID) + '\n')
    i = 0
    for line in tmp :
        if data.labels.patternID is not None :
            outfile.write(data.labels.patternID[i])
        i += 1
        outfile.write(line)
    os.remove(tmpfile)

def averageEntry(fileName, ignoreDiagonal = True, delim = None) :

    s = 0
    numEntries = 0
    file = open(fileName)
    i = 0
    for line in file :
        tokens = line.split(delim)
        for token in tokens :
            try :
                val = float(token)
                s += val
                numEntries += 1
            except :
                pass
        i += 1
    return s / numEntries



def combineKernels(ker1file, ker2file, kerOutFile, operation = 'add', **args) :
    """combine two kernels by either adding or multiplying them.
    In the case of addition the resulting kernel is of the form:
    K_out(i,j) = weight * K1(i,j) + (1-weight) * K2(i,j)
    where the default weight is 0.5
    In the case of multiplication the resulting kernel is:
    K_out(i,j) = (const1 + K1(i,j)) * (const2 + K2(i, j))
    where const1 and const2 are 0 by default.

    Notes:  It is assumed that the kernels have the same size and the ids
    are in the same order (an exception is raised if this is not satisfied).

    :Parameters:
      - `operation` - which operation to perform between the kernels; it is
        a string with supported values 'add' or 'multiply' (add by default)

    :Keywords:
      - `weight` - weighting of kernels for kernel addition
      - `const1,const2` - additive factor in case of kernel multiplication
    """

    weight = 0.5
    if 'weight' in args :
        weight = args['weight']
    const1 = 0
    if 'const1' in args :
        const1 = args['const1']
    const2 = 0
    if 'const2' in args :
        const2 = args['const2']
    import misc
    delim1 = misc.getDelim(ker1file)
    delim2 = misc.getDelim(ker2file)
    ker1 = open(ker1file)
    ker2 = open(ker2file)
    kerOut = open(kerOutFile, 'w')

    # check if kernel is in gist format
    line1 = ker1.readline()
    try :
        float(line1.split(delim1)[-1])
    except :
        line1 = ker1.readline()
    line2 = ker2.readline()
    try :
        float(line2.split(delim2)[-1])
    except :
        line2 = ker2.readline()        

    # check if there's a pattern id:
    firstToken = 0
    try :
        float(tokens1[0])
    except :
        firstToken = 1
            
    while len(line1) > 0 :
        tokens1 = line1.split(delim1)
        tokens2 = line2.split(delim2)
        if firstToken > 0 :
            if tokens1[0] != tokens2[0] :
                print tokens1[0], tokens2[0]
                raise ValueError, 'kernels do not have the same ids'
            kerOut.write(tokens1[0] + delim1)
        if operation == 'add' :
            outTokens = [str(float(tokens1[i]) * weight +
                             float(tokens2[i]) * (1-weight))
                         for i in range(firstToken, len(tokens1))]
        else :
            outTokens = [str((const1 + float(tokens1[i])) *
                             (const2 + float(tokens2[i])))
                         for i in range(firstToken, len(tokens1))]            
        kerOut.write(delim1.join(outTokens) + '\n')
        line1 = ker1.readline()
        line2 = ker2.readline()

def sortKernel(kernelInFile, kernelOutFile, format = 'gist', **args) :
  """
  sort a kernel matrix according to its pattern ID

  :Parameters:
    - `kernelInFile` - the kernel input file name
    - `kernelOutFile` - the output file name
    - `format` - whether to output the kernel in gist format

  :Keywords:
    - `delim` - the field delimiter (default = tab)
  """
  
  from PyML.containers import KernelData
  kdata = KernelData(kernelInFile)
  idDict = misc.list2dict(kdata.labels.patternID, range(len(kdata)))
  ids = kdata.labels.patternID[:]
  ids.sort()
  delim = '\t'
  if 'delim' in args :
    delim = args['delim']
  kernelFile = open(kernelOutFile, 'w')
  if format == 'gist' :
    kernelFile.write(kernelOutFile + delim + delim.join(ids) + '\n')

  for id1 in ids :
    kernelFile.write(id1 + delim)
    tokens = [str(kdata.kernel.eval(kdata, idDict[id1], idDict[id2]))
              for id2 in ids]
    kernelFile.write(delim.join(tokens) + '\n')
  

def commonKernel(kernelFile1, kernelFile2, kernelOutFileName1, kernelOutFileName2) :
    
    delim = ' '
    from datafunc import KernelData
    import misc
    kdata1 = KernelData(kernelFile1)
    kdata2 = KernelData(kernelFile2)
    print 'loaded data'
    ids = misc.intersect(kdata1.labels.patternID, kdata2.labels.patternID)
    ids.sort()
    idDict1 = misc.list2dict(ids)

    if len(ids) != len(kdata1) :
        kernelOutFile1 = open(kernelOutFileName1, 'w')
        idDict = {}
        for i in range(len(kdata1)) :
            if kdata1.labels.patternID[i] in idDict1 :
                idDict[kdata1.labels.patternID[i]] = i
        for id1 in ids :
            print id1
            kernelOutFile1.write(id1 + delim)
            tokens = [str(kdata1.kernel.eval(kdata1, idDict[id1], idDict[id2]))
                      for id2 in ids]
            kernelOutFile1.write(delim.join(tokens) + '\n')
            
    if len(ids) != len(kdata2) :
        kernelOutFile2 = open(kernelOutFileName2, 'w')
        idDict = {}
        for i in range(len(kdata2)) :
            if kdata2.labels.patternID[i] in idDict1 :
                idDict[kdata2.labels.patternID[i]] = i
        for id1 in ids :
            print id1
            kernelOutFile2.write(id1 + delim)
            tokens = [str(kdata2.kernel.eval(kdata2, idDict[id1], idDict[id2]))
                      for id2 in ids]
            kernelOutFile2.write(delim.join(tokens) + '\n')

            
def expandKernel(inKernelFile, referenceKernelFile, outKernelFile, **args) :

    """
    Given a kernel matrix that might have missing entries, fill those as 0
    on the basis of the patterns in a reference kernel (it is checked that
    the reference kernel is sorted).
    
    :Parameters:
      - `inKernelFile` - input kernel file name
      - `referenceKernelFile` - file name for the reference kernel
      - `outKernelFile` - file name to output expanded kernel
    """

    if 'format' in args :
        format = args['format']
    else :
        format = 'gist'
    delim = '\t'

    from datafunc import KernelData
    import misc
    import numpy

    inKernel = KernelData(inKernelFile)
    refKernel = KernelData(referenceKernelFile)
    print 'loaded data'
    ids = refKernel.labels.patternID[:]
    ids.sort()
    if ids != refKernel.labels.patternID :
        raise ValueError, 'reference kernel not sorted'
    
    idDict = misc.list2dict(inKernel.labels.patternID)
    outKernel = open(outKernelFile, 'w')
    if format == 'gist' :
        outKernel.write(outKernelFile + delim)
        outKernel.write(delim.join(ids) + '\n')
    
    for i in range(len(refKernel)) :
        outKernel.write(id1 + delim)
        for j in range(len(refKernel)) :
            values = numpy.zeros(len(refKernel), numpy.float_)
            if ids[i] in idDict and ids[j] in idDict :
                values[j] = inKernel.kernel.eval(inKernel,
                                                 idDict[ids[i]],idDict[ids[j]])
            tokens = [str(value) for value in values]
            outKernel.write(delim.join(tokens) + '\n')

def showKernel(dataOrMatrix, fileName = None, useLabels = True, **args) :
 
    labels = None
    if hasattr(dataOrMatrix, 'type') and dataOrMatrix.type == 'dataset' :
	data = dataOrMatrix
	k = data.getKernelMatrix()
	labels = data.labels
    else :
	k = dataOrMatrix
	if 'labels' in args :
	    labels = args['labels']

    import matplotlib

    if fileName is not None and fileName.find('.eps') > 0 :
        matplotlib.use('PS')
    from matplotlib import pylab

    pylab.matshow(k)
    #pylab.show()

    if useLabels and labels.L is not None :
	numPatterns = 0
	for i in range(labels.numClasses) :
	    numPatterns += labels.classSize[i]
	    #pylab.figtext(0.05, float(numPatterns) / len(labels), labels.classLabels[i])
	    #pylab.figtext(float(numPatterns) / len(labels), 0.05, labels.classLabels[i])
	    pylab.axhline(numPatterns, color = 'black', linewidth = 1)
	    pylab.axvline(numPatterns, color = 'black', linewidth = 1)
    pylab.axis([0, len(labels), 0, len(labels)])
    if fileName is not None :
        pylab.savefig(fileName)
	pylab.close()

            
def sortKernel2(kernelInFile, kernelOutFile, ids, format = 'gist', **args) :
  """
  sort a kernel matrix according to the given list of ids

  :Parameters:
    - `kernelInFile` - the kernel input file name
    - `kernelOutFile` - the output file name
    - `format` - whether to output the kernel in gist format

  :Keywords:
    - `delim` - the field delimiter (default = tab)
  """
  
  from PyML.containers import KernelData
  kdata = KernelData(kernelInFile)
  K = kdata.getKernelMatrix()
  idDict = misc.list2dict(ids, range(len(ids)))

  delim = '\t'
  if 'delim' in args :
      delim = args['delim']
  kernelFile = open(kernelOutFile, 'w')
  if format == 'gist' :
    kernelFile.write(kernelOutFile + delim + delim.join(ids) + '\n')

  for id1 in ids :
    kernelFile.write(id1 + delim)
    tokens = [str(K[idDict[id1]][idDict[id2]]) for id2 in ids]
    kernelFile.write(delim.join(tokens) + '\n')
  

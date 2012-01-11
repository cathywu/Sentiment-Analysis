
import numpy
import random
import os
import myio

__docformat__ = "restructuredtext en"


def my_import(name) :

    mod = __import__(name)
    components = name.split('.')
    for comp in components[1:]:
        mod = getattr(mod, comp)

    return mod

def is_nan(x) :

    return (type(x) == type(float) and x !=x)

def subseteq(A, B) :

    if type(B) != type({}) :
        Bdict = list2dict(B)
    else:
        Bdict = B

    all = 1
    for a in A :
        if a not in Bdict :
            all = 0

    return all
    

def setminus(A, B):

    Bdict = {}
    for b in B :
        Bdict[b] = 1

    result = []
    for a in A :
        if not Bdict.has_key(a) :
            result.append(a)

    return result

def unique(A) :
    '''return the unique elements of a list'''
    
    Adict = {}
    for a in A :
        Adict[a] = 1

    return Adict.keys()

def consecutiveUnique(A, B = None) :

    if len(A) == 0 :
        if B is None :
            return []
        else :
            return ([], [])
    
    resultA = [A[0]]
    if B is not None :
        resultB = [B[0]]
        
    for i in range(1,len(A)) :
        if A[i] != A[i-1] :
            resultA.append(A[i])
            if B is not None :
                resultB.append(B[i])

    if B is None :
        return resultA
    else :
        return (resultA, resultB)

def listEqual(A, B) :
    '''Determine if the lists A and B contain exactly the same elements
    (the lists are treated as multisets)'''
    
    A.sort()
    B.sort()

    allequal = 1
    if len(A) != len(B) : return 0

    for i in range(len(A)) :
        if A[i] != B[i] : return 0

    return 1

def setEqual(A, B) :

    Adict = list2dict(A)
    for b in B :
        if b not in Adict :
            return 0
        else :
            Adict[b] = 0

    if sum(Adict.values()) > 0 :
        return 0
    else :
        return 1


def intersect(A, B) :

    S = []
    for b in B :
        if b in A and not b in S :
            S.append(b)

    return S

def intersectDicts(A, B) :

    Anew = {}
    Bnew = {}
    for key in A.keys() :
        if key in B:
            Anew[key] = A[key]
            Bnew[key] = B[key]

    return Anew, Bnew
    
    
def intersectIndices(A, B) :

    I = []
    Bdict = {}
    for b in B :
        Bdict[b] = 1
        
    for i in range(len(A)) :
        if Bdict.has_key(A[i]) :
            I.append(i)

    return I

def intersectSorted(A, B) :

    S = []
    i = 0
    j = 0
    while i < len(A) and j < len(B) :
        if A[i] == B[j] :
            S.append(A[i])
            i += 1
            j += 1
        elif A[i] > B[j] :
            j += 1
        else :
            i += 1

    return S


def union(A, B) :

    Adict = list2dict(A)

    # make sure we have a list:
    if type(A) != type([]) :
        U = [a for a in A]
    else :
        U = A[:]
    for b in B :
        if b not in Adict :
            U.append(b)

    return U


def mergeSorted(A, B) :

    S = []
    i = 0
    j = 0
    while i < len(A) and j < len(B) :
        if A[i] == B[j] :
            S.append(A[i])
            i += 1
            j += 1
        elif A[i] > B[j] :
            S.append(B[j])
            j += 1
        else :
            S.append(A[i])
            i += 1

    if i < len(A) :
        S = S + A[i:]
    if j < len(B) :
        S = S + B[j:]
        
    return S


def invertDict(A) :

    Ai = {}
    for key in A :
        Ai[A[key]] = key

    return Ai
    
def invert(A) :

    Adict = {}
    for i in range(len(A)) :
        Adict[A[i]] = i

    return Adict


def majority(A) :

    counts = {}
    for a in A :
        if not counts.has_key(a) :
            counts[a] = 1
        else :
            counts[a] += 1

    M = 0
    for key in counts.keys() :
        if counts[key] > M :
            maj = key
            M = counts[key]

    return maj

def idSubList(A, ids, idlist, *options) :
    '''Take a sublist of a list where each member has an id
    the sublist is taken according to the given sublist of ids
    these indicate either ids to take or ids to remove (default
    behavior is to take the ids in idlist, use the option "remove"
    for the other behavior)
    '''
    
    subA = []
    if 'remove' in options :
        idlist = setminus(ids, idlist)
    idlist = list2dict(idlist)
    for i in range(len(ids)) :
        if ids[i] in idlist :
            subA.append(A[i])

    return subA


def subList(A, I, J = None) :
    '''return a sublist of a list
    INPUT
    A - list, list of lists, or a list of strings
    I - subset of "rows" (first index) to take
    J - subset of "columns" (second index) to take (optional)
    returns A[i] for i in I
    or A[i][j] for i in I and j in J if J is given
    '''

    if J is None :
        return [A[i] for i in I]
    elif type(A[0]) == type([]) :
        print 1
        return [[A[i][j] for j in J] for i in I]
    elif type(A[0]) == type('') :
        result = []
        for i in I :
            result.append(''.join([A[i][j] for j in J]))
        return result
    else :
        print 'wrong type of input'
        

def list2dict(A, val = None) :
    '''convert a list to a dictionary
    If a value list is not given, then the value 1 is associated with
    each element in the dictionary
    to assign each element its position in the list use:
    list2dict(A, range(len(A)))
    '''

    if type(A) == type({}) : return A
    
    D = {}

    if val is None :
        for a in A :
            D[a] = 1
    elif len(A) == len(val) :
        for i in range(len(A)) :
            D[A[i]] = val[i]
    else :
        print 'list and value list do not have the same length'
            
    return D


def dictCount(A) :

    D = {}
    for a in A :
        if a in D :
            D[a] += 1
        else :
            D[a] = 1

    return D

    
def emptyLOL(n) :

    return [[] for i in range(n)]


def matrix(shape, value = None) :

    return [[value for j in range(shape[1])] for i in range(shape[0])]

## def transpose(matrix) :

##     m = len(matrix)
##     n = len(matrix[0])
##     A = matrix((n,m))
##     for i in range(m) :
##         for j in range(n) :
##             A[j][i] = matrix[i][j]

##     return A

def LOD(n) :

    e = []
    for i in range(n) :
        e.append({})

    return e
        

def inverseCumulative(x, v) :

    x = numpy.asarray(x)    

    num = int(len(x) * v)

    x = numpy.sort(x)

    return x[num]

    
def translate(id, idList) :

    idDict = list2dict(idList, range(len(idList)))

    id2 = []

    for elem in id :
        id2.append(idDict[elem])

    return id2


def count(A) :
    '''count the number of occurrences of each element in a list'''

    counts = {}

    for a in A :
        if a in counts :
            counts[a] += 1
        else :
            counts[a] = 1

    return counts


class Container (object) :

    def __init__(self, attributeDict = {}) :

        for attribute in attributeDict :
            self.__setattr__(attribute, attributeDict[attribute])

    def __repr__(self) :

        maxLength = 10
        rep = ''
        for attribute in self.__dict__ :
            try :
                l = len(self.__getattribute__(attribute))
                if l > maxLength :
                    rep += 'length of ' + attribute + ' ' + str(l) + '\n'
                else :
                    rep += attribute + ' ' + str(self.__getattribute__(attribute)) + '\n'
            except :
                rep += attribute + ' ' + str(self.__getattribute__(attribute)) + '\n'

        return rep[:-1]

    def addAttributes(self, object, attributes) :

        for attribute in attributes :
            if hasattr(object, attribute) :
                self.__setattr__(attribute, object.__getattribute__(attribute))


def extractAttribute(l, attribute) :

    if type(l) == type({}) :
        out = {}
        for k in l.keys() :
            out[k] = getattr(l[k], attribute)

    elif type(l) == type([]) :
        out = []
        for elem in l :
            out.append(getattr(elem, attribute))

    return out
                       
        
def split(s, delim) :

    if delim == ' ' :
        return s.split()
    else :
        return s.split(delim)


def flat(A) :

    outlist = []
    for a in A :
        outlist.extend(a)

    return outlist


def transpose(A) :

    if type(A[0]) == type([]) or type(A[0]) == type((1)) :

        return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

    elif type(A[0]) == type('') :

        m = len(A)
        n = len(A[0])

        B = []

        for i in range(n) :
            B.append(A[0][i])
            for j in range(1,m) :
                B[i] += A[j][i]

        return B

    else :
        raise ValueError, 'wrong type of Input'


def dictProjection(A, B) :

    Aprojected = {}
    Bprojected = {}
    for a in A :
        if a in B :
            Aprojected[a] = A[a]
            Bprojected[a] = B[a]

    return Aprojected,Bprojected

def randsubset(length, subsetLength) :
    '''returns a random subset of range(length) of size subsetLength'''

    if type(length) == type(1) :
        I = range(length)
        random.shuffle(I)
        I = I[:subsetLength]
        I.sort()
        return I
    else :
        raise ValueError, 'wrong type of argument'

class Null :
    """
    Null objects always and reliably, do nothing.
    """

    def __init__(self, *args, **kwargs) : pass
    def __call__(self, *args, **kwargs) : return self
    def __repr__(self) : return "Null()"
    def __nonzero__(self) : return 0

    def __getattr__(self, name) : return self
    def __setattr__(self, name, value) : return self
    def __delattr__(self, name) : return self

def sortDict(dict) :
    '''sort the values and keys of a dictionary
    assumes values are numeric'''
    
    values = dict.values()
    keys = dict.keys()

    ranking = numpy.argsort(values)
    sorted = numpy.sort(values)

    sortedKeys = [keys[ranking[i]] for i in range(len(keys))]

    return sorted, sortedKeys

def dict2array(dict) :

    a = numpy.zeros(max(dict.keys()) + 1, numpy.float_)
    for key in dict :
        a[key] = dict[key]

    return a

def argmax(A) :
    '''returns the indices of the maximum element of a two dimensional matrix'''

    if len(numpy.shape(A)) != 2 :
        raise ValueError, 'wrong shape for matrix'

    (m,n) = numpy.shape(A)

    maxElem = numpy.argmax(A.flat)
    
    return divmod(maxElem, n)


def splitFileName(fileName) :
    
    fileName = os.path.abspath(fileName)
    (directory, fileName) = os.path.split(fileName)
    (base, ext) = os.path.splitext(fileName)
    directory += '/'
    
    return (directory, base, ext)

def unravel(l) :
    
    r = []
    for element in l :
        r.extend(element)

    return r

def findDelim(handleOrName) :

    commentChars = ['%', '#']
    if type(handleOrName) == type('') :
        fileHandle = myio.myopen(handleOrName)
    else :
        fileHandle = handleOrName
    pos = fileHandle.tell()
    line = fileHandle.readline()
    while line[0] in commentChars :
        line = fileHandle.readline()
    line = fileHandle.readline()
    delims = [',', ' ', '\t']
    length = 0
    for delim in delims :
        l = len(line.split(delim))
        if l >= length :
            delimiter = delim
            length = l
    fileHandle.seek(pos)
    
    return delimiter

getDelim = findDelim

def adjacencyMatrix(fileName) :

    file = open(fileName)
    delim = findDelim(file)

    E = {}
    for line in file :
        tokens = split(line, delim)
        try :
            v1 = int(tokens[0])
            v2 = int(tokens[1])
        except :
            v1 = tokens[0]
            v2 = tokens[0]
        if v1 not in E : E[v1] = {}
        if v2 not in E : E[v2] = {}
        E[v1][v2] = 1
        E[v2][v1] = 1

    return E

def getArch() :
    
    (input, output) = os.popen4('arch')
    arch = output.readline()
    #arch = arch.strip()
    return arch.strip()

class MyList (list) :

    def __init__(self, arg1 = None, arg2 = None, arg3 = None, *options, **args) :

        list.__init__(self)

    def append(self, arg, arg1 = None, arg2 = None, arg3 = None) :

        list.append(self, arg)

    def appendPrediction(self, arg1, arg2, arg3) :

	list.append(self, arg)


    def computeStats(self) :

        pass

class DecisionFuncResults (object) :

    def __init__(self, arg1 = None, arg2 = None, arg3 = None, *options, **args) :

	self.decisionFunc = []

    def appendPrediction(self, arg1, arg2, arg3) :

	self.decisionFunc.append(arg1[0])


    def computeStats(self) :

        pass
    
def mysetattr(obj, attribute, value) :

    tokens = attribute.split('.')
    if len(tokens) == 1 :
        setattr(obj, attribute, value)
    else :
        childObj = getattr(obj, tokens[0])
        mysetattr(childObj, '.'.join(tokens[1:]), value)
    
def isString(var) :
    """
    determine whether a variable is a string
    """
    
    return type(var) in (str, unicode)

def set_attributes(x, values, defaults = None) :

    if isinstance(x, dict):
        x.update(defaults)
        x.update(values)
    else :
        if defaults is None : defaults = values
        for attribute in defaults :
            if attribute in values :
                setattr(x, attribute, values[attribute])
            else :
                setattr(x, attribute, defaults[attribute])

def update(x, **entries):
    """Update a dict or an object with according to entries.
    >>> update({'a': 1}, a=10, b=20)
    {'a': 10, 'b': 20}
    >>> update(Struct(a=1), a=10, b=20)
    Struct(a=10, b=20)
    """
    if isinstance(x, dict):
        x.update(entries)   
    else :
        for attribute in entries :
            setattr(x, attribute, entries[attribute])


def timer(fn, *args):
    """Time the application of fn to args. Return (result, seconds)."""

    import time
    start = time.clock()
    return fn(*args), time.clock() - start

def get_defaults(defaults, args, varNames) :

    returnList = []
    for name in varNames :
        if name not in defaults :
            raise ValueError, 'argument mission in defaults'
        if name in args :
            returnList.append(args[name])
        else :
            returnList.append(defaults[name])
    return returnList


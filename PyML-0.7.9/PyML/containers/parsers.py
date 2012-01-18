
import os
from PyML.utils import misc,myio
from PyML.base.pymlObject import PyMLobject

__docformat__ = "restructuredtext en"

class Parser (PyMLobject) :

    '''A parser class to read datasets from a file.
    Each parser support the following interface:
    Constructor - pass a file name / file handle and information on which
    pattern/classes/features to read from the file
    check - checks whether the file conforms to the format read by the parser
    scan - scan the file and make the _address variable that lists the positions
    in the file of all the patterns that need to be read
    next - read the next pattern (after calling the __iter__ method)
    '''

    commentChar = ['%', '#']
    
    def __init__(self, file, **args) :

        if type(file) == type('') :
            if not os.path.exists(file) :
                raise ValueError, "file does not exist at %s" % file
            self._fileHandle = myio.myopen(file)
            #self._fileHandle = open(file)
        else :
            self._fileHandle = file

        if 'classes' in args :
            self.classesToRead = args['classes']
        else :
            self.classesToRead = []

        if 'patterns' in args :
            self.patternsToRead = args['patterns']
        else :
            self.patternsToRead = None

        if 'features' in args :
            self.featuresToRead = args['features']
        else :
            self.featuresToRead = []


    def check(self) :

        pass

    def scan(self) :

        pass

    def __iter__(self) :

        self._addressIterator = iter(self._address)
        return self


    def __len__(self) :
        '''how many patterns are read'''
        
        return len(self._address)
    

    def next(self) :

        pass

    def skipComments(self) :

        pos = 0
        line = self._fileHandle.readline()
        while line[0] in self.commentChar :
            pos += len(line)
            line = self._fileHandle.readline()
        
        return line, pos

class SparseParser (Parser) :

    '''A class for parsing sparse data'''

    def __init__(self, file, **args) :

        Parser.__init__(self, file, **args)
        self.sparsify = False
        if 'sparsify' in args :
            self.sparsify = args['sparsify']

    def check(self) :

        self._fileHandle.seek(0)
        line,pos = self.skipComments()
        return len(line.split(':')) > 1

    def readLabels(self) :
        
        self._fileHandle.seek(0)
        patternID = None
        L = []
        
        line,pos = self.skipComments()
        # determine if the dataset has IDs :
        patternIDflag = (line.find(",") != -1)
        if patternIDflag :
            patternID = []        
        # make sure there are labels :
        tokens = line.split(',')[-1].split()
        if len(tokens) == 0 or tokens[0].find(':') >= 0 :
            raise ValueError, "unlabeled data"
        
        while line :
            
            if patternIDflag:
                (patID, line) = line.split(",")
                patternID.append(patID)
            L.append(line.split()[0])

            line = self._fileHandle.readline()            

        return L,patternID

    def scan(self) :

        self._fileHandle.seek(0)
        patternID = None
        self._featureID = []
        
        address = []

        line, pos = self.skipComments()

        # determine if the dataset has IDs :
        patternIDflag = (line.find(",") != -1)
        if patternIDflag :
            patternID = []

        # determine if the dataset has labels or not :
        tokens = line.split(',')[-1].split()
        if len(tokens) == 0 or tokens[0].find(':') >= 0 :
            L = None
            labeledData = 0
            firstToken = 0
        else :
            L = []
            labeledData = 1
            firstToken = 1

        self._numFeatures = 0
        
        self.integerID = True
            
        i = 0
        featureDict = {}
        foundIntegerID = False
        while line :
            nextPos = pos + len(line)
            if patternIDflag:
                (patID, line) = line.split(",")

            tokens = line.split()
            if labeledData :
                label = tokens[0]
            else :
                label = None
            if not foundIntegerID :
                if labeledData :
                    t = tokens[1:]
                else :
                    t = tokens
                if len(t) > 0 :
                    foundIntegerID = True
                    for token in t :
                        try :
                            int(token.split(':')[0])
                        except :
                            self.integerID = False

            if (label in self.classesToRead or len(self.classesToRead) == 0) :
                if labeledData :
                    L.append(label)
                if patternIDflag :
                    patternID.append(patID)
                address.append(pos)

            pos = nextPos
            line = self._fileHandle.readline()
            i +=1
#            if i % 100 == 0 and i > 0 :
#                print 'scanned',i,'patterns'
        
        self._featureDict = {}
        self._featureDict2 = {}
        self._featureKeyDict = {}
        self._address = address
        self._labeledData = labeledData
        self._labels = L
        self._patternIDflag = patternIDflag
        self._patternID = patternID
        self._firstToken = firstToken


    def __iter__(self) :

        self._addressIterator = iter(self._address)
        
        return self

    def next(self) :

        address = self._addressIterator.next()
        self._fileHandle.seek(address)
        
        line = self._fileHandle.readline()
        if self._patternIDflag:
            (patID, line) = line.split(",")

        tokens = line.split()
        if self._labeledData :
            label = tokens[0]
        else :
            label = None
            
        x = {}
        if len(tokens) > self._firstToken :  # check if this is not a zero vector
            for token in tokens[self._firstToken:] :
                (featureID, featureVal) = token.split(":")
                if self.integerID :
                    featureID = int(featureID)

                uniqueHash = True
                # handle the case where the hash function is not unique:
                if (featureID in self._featureDict2 and
                    self._featureDict2[featureID] != featureID) :
                    uniqueHash = False
                    #XXX
                    for i in range(255) :
                        fid = featureID + '+' + chr(i)
                        if fid not in self._featureDict2 :
                            featureID = fid
                            uniqueHash = True
                if not uniqueHash :
                    raise ValueError, 'non-unique hash'

                if not self.integerID :
                    featureKey = hash(featureID)
                else :
                    featureKey = featureID
                self._featureDict[featureID] = featureKey
                self._featureDict2[featureID] = featureID
                self._featureKeyDict[featureKey] = 1

                if float(featureVal) != 0.0 or not self.sparsify :
                    #x[self._featureDict[featureID]] = float(featureVal)
                    x[featureKey] = float(featureVal)

        return x

    def postProcess(self) :

        if len(self._featureDict.keys()) != len(misc.unique(self._featureDict.values())) :
            print len(self._featureDict.keys()), len(misc.unique(self._featureDict.values()))
            raise ValueError, 'non-unique hash'
        
        featureKeyDict = {}
        featureKey = self._featureDict.values()
        featureKey.sort()
        for i in range(len(featureKey)) :
            featureKeyDict[featureKey[i]] = i
        inverseFeatureDict = misc.invertDict(self._featureDict)
        featureID = [str(inverseFeatureDict[key]) for key in featureKey]

        return featureID, featureKey, featureKeyDict


class CSVParser (Parser):

    """A class for parsing delimited files"""

    attributes = {'idColumn' : None,
                  'labelsColumn' : None,
                  'headerRow' : False}
    
    def __init__(self, file, **args) :

        """
        :Keywords:
          - `headerRow` - True/False depending on whether the file contains a
            header row that provides feature IDs
          - `idColumn` - set to 0 if the data has pattern IDs in the first column
          - `labelsColumn` -  possible values: if there are no patternIDs
            it is either 0 or -1, and if there are patternIDs, 1 or -1
        """

        Parser.__init__(self, file, **args)
	PyMLobject.__init__(self, None, **args)

	if self.labelsColumn == 1 :
	    self.idColumn = 0
        if self.idColumn is None and self.labelsColumn is None :
            self._first = 0
        else :
            self._first = max(self.idColumn, self.labelsColumn) + 1
#        print 'label at ', self.labelsColumn

    def check(self) :
        """very loose checking of the format of the file:
        if the first line does not contain a colon (":") it is assumed
        to be in csv format
        the delimiter is determined to be "," if the first line contains
        at least one comma; otherwise a split on whitespaces is used.
        """
        
        self._fileHandle.seek(0)
        
        if self.headerRow :
            line = self._fileHandle.readline()
        else :
            line,pos = self.skipComments()
        if len(line.split('\t')) > 1 :
            self.delim = '\t'
        elif len(line.split(',')) > 1 :
            self.delim = ','
        else :
            self.delim = None
        #line,pos = self.skipHeader(line,pos)
#        print 'delimiter', self.delim

        # a file that does not contain a ":" is assumed to be in
        # CSV format

        if len(line.split(':')) > 1  : return False
            
        return True
    
    def skipHeader(self, line, pos) :
        """
        check if the file has a first line that provides the feature IDs
        """

        tokens = line[:-1].split(self.delim)
        if self.labelsColumn == -1 :
            self._last = len(tokens) - 1
        else :
            self._last = len(tokens)

	if self.headerRow :
            self._featureID = tokens[self._first:self._last]
            pos += len(line)
            line = self._fileHandle.readline()

        
        return line, pos

    def readLabels(self) :

        self._fileHandle.seek(0)

        L = []
        patternID = []
        
        line,pos = self.skipComments()
        line, pos = self.skipHeader(line, pos)        
        tokens = line[:-1].split(self.delim)
        if self.labelsColumn is None :
            if len(tokens) == 2 :
                self.labelsColumn = 1
                self.idColumn = 0
            elif len(tokens) == 1 :
                self.labelsColumn = 0
        
        i = 1
        while line :
            tokens = line[:-1].split(self.delim)
            if self.idColumn is not None :
                patternID.append(tokens[self.idColumn])
            else :
                patternID.append(str(i))
            if self.labelsColumn is not None :
                L.append(tokens[self.labelsColumn])
            line = self._fileHandle.readline()            
            i =+ 1
            
        return L,patternID


    def scan(self) :

        self._fileHandle.seek(0)
        self._featureID = None
        address = []

        line,pos = self.skipComments()
        line, pos = self.skipHeader(line, pos)        
            
        tokens = line.split(self.delim)
        self._patternID = []

        dim = len(tokens) - (self.idColumn is not None) - \
              (self.labelsColumn is not None)

        self._labels = None
        if self.labelsColumn is not None :
            self._labels = []

        i = 0
        while line :
            address.append(pos)
            pos += len(line)
            line = self._fileHandle.readline()
            i +=1
#           if i % 1000 == 0 and i > 0 :
#                print 'scanned',i,'patterns'

        self._address = address
        if self._featureID is None :
            self._featureID = [str(i) for i in range(dim)]


    def next(self) :
        
        address = self._addressIterator.next()
        self._fileHandle.seek(address)
        
        line = self._fileHandle.readline()
        tokens = line[:-1].split(self.delim)
        x = [float(token) for token in tokens[self._first:self._last]]
        if self.labelsColumn is not None :
            self._labels.append(tokens[self.labelsColumn])
        if self.idColumn is not None :
            self._patternID.append(tokens[self.idColumn])

        return x

    def postProcess(self) :

        featureKey = [hash(id) for id in self._featureID]
        featureKeyDict = {}
        for i in range(len(featureKey)) :
            featureKeyDict[featureKey[i]] = i

        return self._featureID, featureKey, featureKeyDict
    
def parserDispatcher(fileHandle, **args) :

    if 'hint' in args :
        hint = args['hint']
        if hint == 'sparse' :
            return SparseParser(fileHandle, **args)
        elif hint == 'csv' :
            p = CSVParser(fileHandle, **args)
            p.check()
            #print 'returning a csv parser'
            return p
        
    p = SparseParser(fileHandle, **args)
    if p.check() :
        return p

    p = CSVParser(fileHandle, **args)
    if p.check() :
        return p

    raise ValueError, 'file does not match existing parsers'


def test(fileName) :

    p = SparseParser(fileName)

    print 'p.check:',p.check()

    p.scan()



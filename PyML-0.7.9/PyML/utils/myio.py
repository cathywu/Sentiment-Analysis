
import os
import string
import numpy
import math
import misc
import tempfile

def load(fileName) :

    import cPickle
    
    if not os.path.exists(fileName) :
        raise ValueError, "file does not exist at %s" % fileName

    fileHandle = open(fileName, "r")

    u = cPickle.Unpickler(fileHandle)
    object = u.load()
    fileHandle.close()
    return object

def xmlLoad(fileName) :

    import gnosis.xml.pickle as xml_pickle

    if not os.path.exists(fileName) :
        raise ValueError, "file does not exist at %s" % fileName

    file = open(fileName)

    return xml_pickle.load(file)

def xmlSave(object, fileName) :

    import gnosis.xml.pickle as xml_pickle

    file = open(fileName, 'w')
    xml_pickle.dump(object, file)
    file.close()

def save(object, fileName, binary = 1) :

    import cPickle

    fileHandle = open(fileName, "w")
    p = cPickle.Pickler(fileHandle)
    p.dump(object)
    fileHandle.close()


def csvwrite(a, fileName, delim = ',') :
    '''write an array to a file in csv (comma-delimited) format
    '''
    
    fileHandle = open(fileName,"w")

    if type(a[0]) == type('') or type(a[0]) == type(1) or type(a[0]) == type(1.0) \
       or (type(a).__name__ == 'array' and len(numpy.shape(a)) == 1)  :
        for i in range(len(a)) :
            fileHandle.write(str(a[i]) + '\n')
        fileHandle.close()
        return
    
    for i in range(len(a)) :
        out = ""
        for j in range(len(a[i])) :
            out += str(a[i][j])
            if j < len(a[i]) - 1 :
                out = out + delim
        out += '\n'
        fileHandle.write(out)

    fileHandle.close()

def csvread(fileName, delim = ',') :
    '''read a character array from a file in csv format'''
    import misc
    
    fileHandle = open(fileName, "r")

    line = fileHandle.readline()

    if delim == ' ' : delim = None
    data = misc.emptyLOL(len(line.split(delim)))
    dim = len(data)
    
    while 1 :
        line = line[:-1]
        fields = line.split(delim)
        if len(fields) != dim :
            print 'badline:', line
        for i in range(dim) :
            data[i].append(fields[i])

        line=fileHandle.readline()

        if not line : break

    if len(data) == 1 : data=data[0]
    
    return data

def csvreadArray(fileName, type = 'float') :
    '''read an array from a file in csv format into a numpy array'''
    
    return dlmreadArray(fileName, ',', type)


def dlmreadArray(fileName, delimiter = ' ', type = 'float') :
    '''read an array from a delimited file into a numpy array.
    all lines are assumed to have the same number of columns'''

    commentChar = ['#', '%']
    fileHandle = open(fileName, "r")
    if delimiter is ' ' : delimiter = None
    n = 0
    for line in fileHandle :
        if line[0] in commentChar : continue
        n += 1
        if n == 1 :
            d = len(line.split(delimiter))

    fileHandle.close()
    fileHandle = open(fileName, "r")

    if type == 'float' :
        arrayType = numpy.float
    elif type == 'int' :
        arrayType = numpy.int
    else :
        raise ValueError, 'Wrong type of array'
    if d == 1 :
        X = numpy.zeros(n, arrayType)
    else :
        X = numpy.zeros((n,d), arrayType)
        
    i=0
    print d,n
    for line in fileHandle :
        if line[0] in commentChar : continue        
        #line = line[:-1]
        fields = line.split(delimiter)
        if len(fields) != d :
            print 'badline:', line
        if d == 1 :
            X[i] = float(fields[0])
        else :
            for j in range(d) :
                X[i][j] = float(fields[j])
        i += 1

    fileHandle.close()

    return X


def tableWrite(t, fileName=None, **args) :
    '''Output a table out of a list of lists; elements number i
    of each list form row i of the table
    Usage :
    tableWrite((list1,list2...)) - write table to stdout
    tableWrite((list1,list2...), fileName) - write table to file
    '''

        
    import sys

    if fileName is not None :
        fileHandle = open(fileName, "w")
    else :
        fileHandle = sys.stdout

    if 'headings' in args :
        headings = args['headings']
    else :
        headings = None
        
    d = len(t)
    n = len(t[0])
    print d,n
    maxlen=numpy.zeros(d)

    if headings != None :
        assert len(headings) == d
        
    for i in range(n) :
        for j in range(d) :
            if type(t[j][i]) == type(1.0) :
                s = "%f" % t[j][i]
            else :
                s = str(t[j][i])
            if len(s) > maxlen[j] :
                maxlen[j] = len(s)

    
    if headings != None :
        for j in range(d) :
            if len(headings[j]) > maxlen[j] :
                maxlen[j] = len(headings[j])
            print >> fileHandle, "%s" % string.center(headings[j], maxlen[j]),
        print >> fileHandle
            
    for i in range(n) :
        for j in range(d) :
            
            if type(t[j][i]) == type("") :
                print >> fileHandle, "%s" % string.ljust(t[j][i], maxlen[j]),
            elif type(t[j][i]) == type(1) :
                print >> fileHandle, "%s" % string.rjust(str(t[j][i]), maxlen[j]),
            elif type(t[j][i]) == type(1.0) :
                s = "%f" % t[j][i]
                print >> fileHandle, "%s" % string.rjust(s, maxlen[j]),
            else :
                print >> fileHandle, "%s" % ' ' * maxlen[j],
                print "unknown data type"
        print >> fileHandle

def dlmWrite(t, fileName, delim = ',') :
        
    fileHandle = open(fileName, "w")
    d = len(t)

    try :
        n = len(t[0])
    except :
        n = len(t)
        for i in range(n) :
            fileHandle.write(str(t[i]) + '\n')

        fileHandle.close()
        
        return
    
    for i in range(n) :
        for j in range(d) :

            fileHandle.write(str(t[j][i]))

            if j == d-1 :
                fileHandle.write('\n')
            else :
                fileHandle.write(delim)


def writeDict(dict, fileName, delim = ',') :
    '''write a dictionary into a file as a set of pairs of key,value
    '''

    file = open(fileName, 'w')

    for k in dict.keys() :
        file.write(str(k) + delim + str(dict[k]) + '\n')

    file.close()

def subDict(dict, list) :

    newDict = {}
    for l in list :
        if l in dict :
            newDict[l] = dict[l]

    return newDict


def concatenateFiles(fileName1, fileName2, fileName3, delim = ' ') :
    """Horizontal concatenation of of two delimited files into a third file
    the delimiter is a space by default
    """
    

    file1 = open(fileName1)
    file2 = open(fileName2)

    file3 = open(fileName3, 'w')

    while 1 :
        line1 = file1.readline()
        line2 = file2.readline()

        if len(line1) == 0 | len(line2) == 0 :
            break
        
        file3.write(line1[:-1] + delim + line2)

    file1.close()
    file2.close()
    file3.close()
    

def datasetIntersect(datasetSourceName, datasetIntersectName, newDatasetName) :

    '''keep the patterns in the source dataset that appear in the intersect
    dataset
    '''

    idDict = {}
    datasetIntersect = open(datasetIntersectName)
    for line in datasetIntersect :
        idDict[line[:line.find(',')]] = 1

    datasetIntersect.close()
    
    datasetSource = open(datasetSourceName)
    newDataset = open(newDatasetName,'w')

    for line in datasetSource :
        id = line[:line.find(',')]
        if id in idDict :
            newDataset.write(line)

    datasetSource.close()
    newDataset.close()
    

def datasetUnion(datasetName1, datasetName2, newDatasetName) :
    '''assumes that the features in the two datasets have
    different names!'''
    
    dataset1 = open(datasetName1)
    dataset2 = open(datasetName2)
    newDataset = open(newDatasetName, 'w')

    for line1 in dataset1 :
        line2 = dataset2.readline()

        newDataset.write(line1[:-1] + ' ')
        tokens = line2.split()[1:]
        newDataset.write(' '.join(tokens) + '\n')


def makeDataSet(XfileName, labelsFileName, datasetFileName) :
    '''make a sparse format data file out of an unlabeled sparse data file
    and a labels file (comma delimited:  id,label)
    '''
    
    if not os.path.exists(XfileName) :
        raise ValueError, "Xfile does not exist at %s" % XfileName

    if not os.path.exists(labelsFileName) :
        raise ValueError, "labels file does not exist at %s" % LabelsFileName

    labelsFile = open(labelsFileName)

    labels = {}
    for line in labelsFile :
        line = line[:-1]
        if len(line.split()) ==2 :
            (id,label) = line.split()
        else :
            (id,label) = line.split(',')
        labels[id] = label

    print len(labels)
    labelsFile.close()
    
    Xfile = open(XfileName)
    datasetFile = open(datasetFileName, 'w')

    for line in Xfile :
        (id, restOfLine) = line.split(',')
        if labels.has_key(id) :
            datasetFile.write(id + ',' + labels[id] + ' ' + restOfLine)

    Xfile.close()
    datasetFile.close()


def formatMotifData(motifFileName, labelsFileName, svmFormatFileName,
                    motifSpecFile = None) :

    labelsFile = open(labelsFileName, 'r')

    labels = {}
    for line in labelsFile :
        line = line[:-1]
        if len(line.split()) == 2 :
            (ac,label)=line.split()
        else :
            (ac,label) = line.split(',')
        labels[ac] = label
        
    labelsFile.close()        

    if motifSpecFile != None :
        motifSpec
        motifSpec = open(motifSpecFile)
        for line in motifSpecFile :
            line = line[:-1]

    svmFormatFile = open(svmFormatFileName, 'w')
    motifFile = open(motifFileName, 'r')

    for line in motifFile :
        
        tokens = line.split(';')
        ac = tokens[0].split(',')[0]

        if labels.has_key(ac) :
            x = {}
            for token in tokens[1:] :
                featureID = token.split(',')[0]
                if x.has_key(featureID) :
                    x[featureID] += 1
                else :
                    x[featureID] = 1

            print >> svmFormatFile, "%s,%s" % (ac,labels[ac]),
            xKeys = x.keys()
            #xKeys.sort()
            for xKey in xKeys :
                print >> svmFormatFile, "%s:%s" % (xKey,x[xKey]),

            print >> svmFormatFile

    svmFormatFile.close()
    motifFile.close()



def formatMotifX(motifFileName, XfileName, motifSpecFile = None) :


    if motifSpecFile != None :
        motifSpec = open(motifSpecFile)
        for line in motifSpecFile :
            line = line[:-1]

    Xfile = open(XfileName, 'w')
    motifFile = open(motifFileName, 'r')

    for line in motifFile :
        
        tokens = line.split(';')
        ac = tokens[0].split(',')[0]

        x = {}
        for token in tokens[1:] :
            featureID = token.split(',')[0]
            if x.has_key(featureID) :
                x[featureID] += 1
            else :
                x[featureID] = 1

        Xfile.write(ac + ',')
        xKeys = x.keys()
        #xKeys.sort()
        for xKey in xKeys :
            print >> Xfile, "%s:%s" % (xKey,x[xKey]),

        print >> Xfile

    Xfile.close()
    motifFile.close()


def dlmExtract(inFile, outFields, outFile = None, convert = True,
               filterFile = None, filterField = None,
               inDelim = ',', outDelim = ',') :
    
    '''Extract from a delimited file a list of fields to another delimited file
    Input:
    inFile - file name with the input data
    outFields - a list of fields to extract from inFile
    outFile - output file
    convert - whether to convert numeric inputs from strings
    inDelim - the delimiter in the input file
    outDelim - the delimiter for the output file
    '''
    

    inFileHandle = open(inFile)
    if outFile is not None :
        outFileHandle = open(outFile, 'w')
        convert = False

    if outFile is None :
        data = []
    else :
        data = None
        
    if type(outFields) == type(1) :
        outFields = [outFields]

    if filterFile is not None :
        filterDict = {}
        filter = open(filterFile)
        for line in filter :
            filterDict[line[:-1]] = 1
        filterFile.close()
        
    for line in inFileHandle :
        line = line[:-1]

        fields = line.split(inDelim)

        if filterFile is None or fields[filterField] in filterDict :

            out = []
            for i in outFields :
                if convert :
                    try :
                        out.append(float(fields[i]))
                    except :
                        out.append(fields[i])
                else :
                    out.append(fields[i])
            if outFile is None :
                data.append(out)
            else :
                outFileHandle.write(outDelim.join(out) + '\n')

    inFileHandle.close()
    return data

def countLines(fileName) :

    if not os.path.exists(fileName) :
        raise ValueError, "file does not exist at %s" % fileName

    file = open(fileName)

    numLines = 0
    for line in file:
        numLines += 1

    file.close()
    
    return numLines


def extractLines(fileName, lines) :
    '''extract the lines given by a list of line numbers in the file'''
    
    file = open(fileName)

    lineDict = misc.list2dict(lines)
    lineNum = 1
    for line in file :
        if lineNum in lineDict :
            print line[:-1]
        lineNum += 1

    file.close()
    

def splitFile(fileName, numFiles) :

    numLines = countLines(fileName)

    if not os.path.exists(fileName) :
        raise ValueError, "file does not exist at %s" % fileName

    file = open(fileName)

    numSplit = int(math.floor(numLines / numFiles))
    
    lineNum = 0
    fileNum = 0
    for line in file :
        if math.fmod(lineNum, numSplit) == 0 :
            fileNum += 1
            try :
                outFile.close()
            except :
                pass
            outFile = open('split' + str(fileNum) + fileName, 'w')

        outFile.write(line)
        lineNum += 1
        
    file.close()
    
    
def log(message, fileName = 'progress.log', openMode = 'a') :

    file = open(fileName, 'a')
    file.write(message)
    file.close()

def isempty(fileName) :

    if not os.path.exists(fileName) :
        return 1
    file = open(fileName)
    if len(file.readlines()) == 0 :
        file.close()
        return 1
    file.close()
    return 0
    

def findDelimiter(fileName) :
    '''guess the delimiter of a delimited file according to the first line
    in the file'''
    
    delimiters = [',', ' ', ';', '\t']

    file = open(fileName)
    line = file.readline()
    file.close()
    
    maxTokens = 0
    maxDelim = ''
    for delim in delimiters :
        splitLen = len(misc.split(line, delim))
        if splitLen > maxTokens :
            maxDelim = delim
            maxTokens = splitLen
    
    return maxDelim


class UndoHandle:
    """A Python handle that adds functionality for saving lines.
    Saves lines in a LIFO fashion.
    Added methods:
    saveline    Save a line to be returned next time.
    peekline    Peek at the next line without consuming it.
    """
    def __init__(self, handle):
        self._handle = handle
        self._saved = []

    def readlines(self, *args, **keywds):
        lines = self._saved + apply(self._handle.readlines, args, keywds)
        self._saved = []
        return lines

    def readline(self, *args, **keywds):
        if self._saved:
            line = self._saved.pop(0)
        else:
            line = apply(self._handle.readline, args, keywds)
        return line

    def read(self, size=-1):
        if size == -1:
            saved = string.join(self._saved, "")
            self._saved[:] = []
        else:
            saved = ''
            while size > 0 and self._saved:
                if len(self._saved[0]) <= size:
                    size = size - len(self._saved[0])
                    saved = saved + self._saved.pop(0)
                else:
                    saved = saved + self._saved[0][:size]
                    self._saved[0] = self._saved[0][size:]
                    size = 0
        return saved + self._handle.read(size)

    def saveline(self, line):
        if line:
            self._saved = [line] + self._saved

    def peekline(self):
        if self._saved:
            line = self._saved[0]
        else:
            line = self._handle.readline()
            self.saveline(line)
        return line

    def tell(self):
        lengths = map(len, self._saved)
        sum = reduce(lambda x, y: x+y, lengths, 0)
        return self._handle.tell() - sum

    def seek(self, *args):
        self._saved = []
        apply(self._handle.seek, args)

    def __getattr__(self, attr):
        return getattr(self._handle, attr)


def removeEmpty(directory) :
    '''remove all files that have size 0 from a directory'''
    
    f = os.popen3("ls -l " + directory)
    lines = f[1].readlines()

    for line in lines[1:] :
        if int(line.split()[4]) == 0 :
            fileName = line.split()[-1]
            print 'removing ',fileName
            os.remove(os.path.join(directory, fileName))

def selectLines(infile, outfile, lines, keepLines = 1) :
    '''write to outfile the lines in infile whose line number is in the
    given list of line numbers'''
    
    infileHandle = open(infile)
    outfileHandle = open(outfile, 'w')
    
    lineDict = misc.list2dict(lines)
    lineNum = 0
    for line in infileHandle :
        lineNum += 1
        if keepLines == 1 :
            if lineNum in lineDict :
                outfileHandle.write(line)
        else :
            if lineNum not in lineDict :
                outfileHandle.write(line)

def return2newLine(inFile, outFile = None) :
    """convert \r to \n (windows file to linux file)"""

    rename = False
    if outFile is None :
        rename = True
        outFile = tempfile.mktemp()
    os.system("tr -d '\r' < " + inFile + " > " + outFile)
    if rename :
        os.rename(outFile, inFile)

def universalNewline(infile, outfile) :

    infileHandle = open(infile, 'U')
    outfileHandle = open(outfile, 'w')
    for line in infileHandle :
        outfileHandle.write(line)

def concatByNum(filePattern, outfileName, directory = '.') :
    """
    filePattern -- a regular expression that looks like: start\d+.dat
    """
    
    files = os.listdir(directory)
    import re
    pattern = re.compile(filePattern)
    numFiles = 1
    for fileName in files :
        if pattern.match(fileName) is not None :
            numFiles += 1

    outfile = open(outfileName, 'w')
    for i in range(1, numFiles) :
        fileName = filePattern.replace('\d+', str(i))
        fileHandle = open(os.path.join(directory, fileName))
        for line in fileHandle :
            outfile.write(line)

def merge(file1, file2, outfile) :
    """merge two files

    file1,file2 - the two files or fie handles to be merged
    outfile - the output file name or handle
    """

    print outfile
    if type(outfile) == type('') :
        outfile = open(outfile, 'w')
    print outfile
    for infile in [file1, file2] :
        if type(infile) == type('') :
            infile = open(infile)
        for line in infile :
            outfile.write(line)

def myopen(fileName, universal = False) :
    """
    returns a file handle to a file which is possibly compressed
    using either gzip or bz2

    myopen tries to open the file as a gzip file or a bz2 file.
    if unsuccessful with either it opens it with the standard open
    command.  if the universal argument is set to True, it opens it
    in 'U' mode that uses universal newline support (i.e. all 
    variations on \n yield \n.  it returns the resulting file handle.
    """


    if not ( os.path.exists(fileName) and os.path.isfile(fileName) ):
        raise ValueError, 'file does not exist at %s' % fileName
    
    import gzip
    fileHandle = gzip.GzipFile(fileName)
    gzippedFile = True
    try :
        line = fileHandle.readline()
        fileHandle.close()
    except :
        gzippedFile = False

    if gzippedFile :
        return gzip.GzipFile(fileName)

    import bz2
    fileHandle = bz2.BZ2File(fileName)
    bzippedFile = True
    try :
        line = fileHandle.readline()
        fileHandle.close()
    except :
        bzippedFile = False

    if bzippedFile :
        return bz2.BZ2File(fileName)

    if universal :
        return open(fileName, 'U')
    else :
        return open(fileName)





__docformat__ = "restructuredtext en"
import random
from PyML.containers import parsers

class Labels (object) :
    """
   	A class that holds the labels of a dataset.

    Attributes::
    
      L - labels provided by the user (strings)
      Y - internal label representation - an integer from 0 to numClasses - 1
      for multilabel problems each pattern has a list of integer labels
      patternID - a list of the ids of each pattern
      classDict - a mapping from L to Y
      classLabels - a list providing the name of class i
      classSize - a list with the number of patterns in each class
      numClasses - the number of classes in the data
    """

    def __init__(self, arg = None, **args) : 

        """
        :Parameters:
          - `arg` - a file name from which to read labels, or a list of labels

        :Keywords:
          - `patternID` - a list of pattern IDs
          - `patterns` - in case of copy construction, which patterns to copy
          - `numericLabels` - a Boolean, indicating whether the labels are
            class labels or numeric values (class labels by default).
          - `positiveClass` - for a two class problem, the identity of the
            positive class.  If the labels are '+1' and '-1' or  '1' and '-1',
            the positive class is detected automatically.
          - `classLabels` - 
          - `forgetClassLabels` - when using copy construction the default behavior
            is to return a Labels object that remembers the set of classes that the
            original object had, even if some classes are no longer represented.
            this keyword allows you to change this behavior, so that the classes of
            the original object are forgotten.
        """

        self.L = None
        self.numericLabels = False
        if 'numericLabels' in args :
            self.numericLabels = args['numericLabels']
        self.positiveClass = None
        if 'positiveClass' in args :
            self.positiveClass = args['positiveClass']
        if 'classLabels' in args :
            self.classLabels = args['classLabels']

        if type(arg) == type('') :
            args['hint'] = 'csv'
            p = parsers.parserDispatcher(arg, **args)
            L, self.patternID = p.readLabels()
        elif arg.__class__ == self.__class__ :
            L, self.patternID = self.copy(arg, **args)
        else :
            if arg is not None :
                L = list(arg)
                if len(L) == 0 : L = None
            else :
                L = None
            if 'patternID' in args :
                self.patternID = args['patternID']
            else :
                self.patternID = range(len(L))

        if L is not None :
            self.processLabels(L)

    def copy(self, other, **args) :

        forgetClassLabels = False
        if 'forgetClassLabels' in args :
            forgetClassLabels = args['forgetClassLabels']
        self.numericLabels = other.numericLabels

        if other.L is not None and not self.numericLabels and not forgetClassLabels :
            self.numClasses = other.numClasses
            self.classLabels = other.classLabels[:]

        if 'patterns' in args :
            patternsToCopy = args['patterns']
        else :
            patternsToCopy = range(len(other))

        if other.L is None :
            return ( None, [other.patternID[p] for p in patternsToCopy] )
        else :
            return ([other.L[p] for p in patternsToCopy],
                    [other.patternID[p] for p in patternsToCopy])


    def extend(self, other, patterns = None) :
        """add to a dataset a list of patterns from another dataset"""
        
        if patterns is None :
            patterns = range(len(other))
        # retrieve labels from other
        for p in patterns :
            if self.L is not None :
                self.L.append(other.L[p])
            if self.patternID is not None :
                self.patternID.append(other.patternID[p])
        if self.L is not None :
            self.processLabels(self.L)
            
        
    def processLabels(self, L, **args) :

        forgetClassLabels = False
        if 'forgetClassLabels' in args :
            forgetClassLabels = args['forgetClassLabels']

        if self.numericLabels or type(L[0]) == type(1.0) :
            self.numericLabels = True
            self.Y = [float(y) for y in L]
            self.L = self.Y
            return
        n = len(L)
        classDict = {}
        Y = []
        if not forgetClassLabels and hasattr(self, 'classLabels') :
            classLabels = self.classLabels
            numClasses = self.numClasses
        else :
            for l in L :
                classDict[l] = 1
            classLabels = classDict.keys()
            classLabels.sort()
            numClasses = len(classLabels)
            if classLabels == ['+1', '-1'] or classLabels == ['1', '-1'] :
                classLabels[0],classLabels[1] = classLabels[1],classLabels[0]
            if self.positiveClass is not None and numClasses == 2 :
                if self.positiveClass not in classLabels :
                    raise ValueError, 'unrecognized positiveClass'
                if classLabels[1] != self.positiveClass :
                    classLabels[0],classLabels[1] = classLabels[1],classLabels[0]
            if 'rest' in classLabels and numClasses == 2 :
                if classLabels[1] == 'rest' :
                    classLabels[0],classLabels[1] = classLabels[1],classLabels[0]

        classDict = {}
        for i in range(len(classLabels)) :
            classDict[classLabels[i]] = i

        classSize = [0 for i in range(numClasses)]
        classes = [[] for i in range(numClasses)]
        for i in range(n) :
            y = classDict[L[i]]
            classSize[y] += 1
            Y.append(y)
            classes[y].append(i)

        self.L = L
        self.Y = Y
        self.classSize = classSize
        self.classLabels = classLabels
        self.classDict = classDict
        self.classes = classes
        self.numClasses = numClasses

    def flip(self, patterns) :

        if self.numClasses != 2 :
            raise ValueError, 'not a two class labeling'
        for p in patterns :
            self.L[p] = self.classLabels[(self.Y[p] + 1) % 2]
        self.processLabels(self.L)

    def __len__ (self) :

        return len(self.patternID)

    def __repr__(self) :

        rep = ''
        if self.L is not None and type(self.L[0]) == type('') :
            rep += 'class Label  /  Size \n'
            for i in range(self.numClasses) :
                rep += ' %s : %d\n' % (self.classLabels[i],self.classSize[i])

        return rep

    def isLabeled(self) :

        if self.L is None :
            return False
        else :
            return True
    
    def save(self, fileName, delim = '\t') :

        fileHandle = open(fileName, 'w')
        for i in range(len(self)) :
            if self.L is not None :
                fileHandle.write(self.patternID[i] + delim + str(self.L[i]) + '\n')
            else :
                fileHandle.write(self.patternID[i] + '\n')

    def convertFromMultiLabel(self) :

        for i in range(len(self.L)) :
            self.L[i] = string.join(self.L[i], ";")


    def mergeClasses(self, classList, newLabel = None) :
        """Merge a list of classes into a new class.

        :Parameters:
        - `classList` - a list of classes to merge; can either provide the
          names of the classes or the index.
        - `newLabel` - the name of the new class (if not given then the label
          is formed by concatenating the names of the merged classes)
          """

        if type(classList[0]) == type(1) :
            classList = [self.classLabels[label] for label in classList]

        if newLabel is None :
            try :
                newLabel = "+".join(classList)
            except :
                newLabel = str(classList)

        for classLabel in classList :
            for p in self.classes[self.classDict[classLabel]] :
                self.L[p] = newLabel
        
        self.processLabels(self.L, forgetClassLabels = True)

    def oneAgainstRest(self, classLabels, className = None) :

        """
        creates a one-against-the-rest labels object

        :Parameters:
          - `classLabels` - a single class name, or a list of class names (string
            or a list of strings)
          - `className` - if given, the new name given to the class

          """

        patternID = self.patternID[:]

        if type(classLabels) == type(1) or type(classLabels) == type('') :
            classLabels = [classLabels]
        if type(classLabels[0]) == type(1) :
            classLabels = [self.classLabels[label] for label in classLabels]

        if className is None :
            className = '+'.join(classLabels)

        newL = []
        for i in range(len(self)):
            if self.L[i] in classLabels :
                newL.append(className)
            else :
                newL.append("rest")

        self.processLabels(newL, forgetClassLabels = True)


def mergeClasses(data, classList, newLabel = None) :
    """Merge a list of classes into a new class.

    :Parameters:
    - `data` - a dataset container
    - `classList` - a list of classes to merge; can either provide the
      names of the classes or the index.
    - `newLabel` - the name of the new class (if not given then the label
      is formed by concatenating the names of the merged classes)

    calls Labels.mergeClasses and returns the dataset with the modified labels

    """

    data.labels.mergeClasses(classList, newLabel)
    data.attachLabels(data.labels)

    return data
        
    
def oneAgainstRest(data, classLabels, className = None) :

    """
    creates a one-against-the-rest dataset/labels object

    :Parameters:
    
      - `data` - a dataset
      - `classLabels` - a single class name, or a list of class names (string
        or a list of strings)
      - `className` - if given, the new name given to the class
      
    Return value::
    
      returns a dataset object where all class labels that are different
      from the given class label/s are converted to a single class
      """
	
    data.labels.oneAgainstRest(classLabels, className)
    data.attachLabels(data.labels)

    return data


def randomLabels(Y) :
    """shuffle the vector Y"""

    Yrand = Y[:]
    random.shuffle(Yrand)

    return Yrand

def eliminateMultiLabeled(data) :

    patterns = [i for i in range(len(data.n))
                if len(data.labels.L[i].split(';')) == 1]

    return data.__class__(data, patterns = patterns)


def eliminateSmallClasses(data, size) :
    """returns a dataset that contains the classes of d that contain
    at least size patterns"""
    
    patterns = []
    for i in range(len(data)) :
        if data.labels.classSize[data.labels.Y[i]] >= size :
            patterns.append(i)

    return d.__class__(d, patterns = patterns)


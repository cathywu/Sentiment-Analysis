
import re
import os
import tempfile
import numpy,math,random
import time
from PyML.utils import misc

rfCodeLocation = os.path.join(os.environ['HOME'], 'random_forests/rf5.f')

from PyML.evaluators import assess
from PyML.classifiers.baseClassifiers import Classifier

def substituteInt(string, number, line) :

    return re.sub(string + "=[0-9]*", string + "=" + str(number), line)

def writeData(data, patterns = None,
              dataFileName = '/tmp/data.train', **args) :
    """write a dataset to a file in a format required by the fortran code"""

    if 'state' in args :
        state = args['state']
    else :
        state = 'training'
    if patterns is None :
        patterns = range(len(data))
    print dataFileName
    dataFile = open(dataFileName, 'w')
    for p in patterns :
        x = data.getPattern(p)
        tokens = []
        if type(x) == type({}) :
            for j in range(data.numFeatures) :
                if data.featureKey[j] not in x :
                    tokens.append('0')
                else :
                    tokens.append(str(x[data.featureKey[j]]))
        else :
            for j in range(data.numFeatures) :
                tokens.append(str(x[j]))
        if data.labels.L is None :
            tokens.append(str(1) + '\n')
        else :
            tokens.append(str(data.labels.Y[p] + 1) + '\n')
        dataFile.write(' '.join(tokens))


def writeCode(rf, codeFileName, executableFileName, data, numPatterns = None,
              **args) :
    """
    modify the fortran RF code according to the properties of the dataset
    """

    if numPatterns is None :
        numPatterns = len(data)
    maxcat = 1
    if 'maxcat' in args :
        maxcat = args['maxcat']

    if 'state' in args :
        state = args['state']
    else :
        state = 'training'
        
    code = open(rfCodeLocation)
    newCode = open(codeFileName, 'w')

    line = code.readline()
    while len(line) > 0 :
        if line.find('DESCRIBE DATA') > 0 :
            line = code.readline()
            line = substituteInt("mdim", data.numFeatures, line)
            if state == 'training' :
                line = substituteInt("nsample0", numPatterns, line)
            else :
                line = substituteInt("nsample0", 1, line)
            line = substituteInt("nclass", rf.labels.numClasses, line)
            line = substituteInt("maxcat", maxcat, line)
            newCode.write(line)
            if state == 'testing' :
                line = code.readline()
                line = substituteInt("ntest", numPatterns, line)
                # labeled test set
                line = substituteInt("labelts", 1, line)                
                newCode.write(line)
        elif line.find('SET RUN PARAMETERS') > 0 :
            line = code.readline()
            line = substituteInt("mtry", rf.numFeatures, line)            
            line = substituteInt("jbt", rf.numTrees, line)
            newCode.write(line)
            line = code.readline()
            line = substituteInt("jclasswt", 1, line)
            newCode.write(line)
        elif line.find('SET IMPORTANCE OPTIONS') > 0 :
            line = code.readline()
            # turn off variable interaction computation
            line = substituteInt("interact", 0, line)       
            newCode.write(line)            
        elif line.find('SET PROXIMITY COMPUTATIONS') > 0 :
            pass
        elif line.find('SET OPTIONS BASED ON PROXIMITIES') > 0 :
            pass
            #line = code.readline()
            #line = substituteInt("nscale", 0, line)
            #line = substituteInt("nprot", 0, line)            
            #newCode.write(line)            
        elif line.find('REPLACE MISSING VALUES') > 0 :
            pass
        elif line.find('GRAPHICS') > 0 :
            pass
        elif line.find('SAVING A FOREST') > 0 :
            line = code.readline()
            line = substituteInt("isavepar", 1, line)
            newCode.write(line)            
        elif line.find('RUNNING A SAVED FOREST') > 0 :
            if state == 'testing' :
                line = code.readline()
                line = substituteInt("irunrf", 1, line)
                newCode.write(line)            
            
        # class weights:
        elif line.find('classwt(1)=151') > 0 :
            for i in range(rf.labels.numClasses) :
                line2 = re.sub(r"\(1\)", "(" + str(i+1) + ")", line)
                if state == 'training' :
                    line2 = re.sub("151",
                                   str(float(len(data)) / data.labels.classSize[i]),
                                   line2)
                else :
                    line2 = re.sub("151", str(1/rf.labels.numClasses), line2)

                newCode.write(line2)
        else :
            newCode.write(line)
        line = code.readline()
        
    newCode.close()
    os.system('g77 ' + codeFileName + ' -o ' + executableFileName)
    

class RF (Classifier) :
    """A feature selector/classifier that wraps Breiman and Cutler's RF code
    OPTIONAL ARGUMENTS :
    numTrees - number of trees to train
    numFeatures - number of features on which to train each tree
    """

    def __init__(self, arg = None, **args) :

        Classifier.__init__(self)
        self.trainingDirectory = None
        self.testingDirectory = None

        self.maxSize = 2e6
        self.numTrees = 200
        self.numFeatures = 0
        if arg.__class__ == self.__class__ :
            other = arg
            self.numTrees = other.numTrees
            self.numFeatures = other.numFeatures
            
        if 'numTrees' in args :
            self.numTrees = args['numTrees']
        if 'numFeatures' in args :
            self.numFeatures = args['numFeatures']
                
    def __repr__(self) :

        rep = '<' + self.__class__.__name__ + ' instance>\n'
        rep += 'numTrees: %d\n' % self.numTrees
        
        return rep

    def __del__(self) :

        self.cleanup()

    def cleanup(self) :
        """remove the directory that was created using training/testing"""
        return
        directory = self.trainingDirectory
        if directory is None : return
        if os.path.exists(directory) :
            os.system('rm -rf ' + directory)
                
    def score(self, data, **args) :
        """returns the z-scores from the RF code
        """

        if data.numFeatures * len(data) > self.maxSize :
            self.score2(data, **args)
        self.train(data, **args)
        scoresFile = open(os.path.join(self.trainingDirectory,
                                       'save-importance-data'))
        scores = numpy.zeros(data.numFeatures, numpy.Float)
        pvals = numpy.zeros(data.numFeatures, numpy.Float)
        for line in scoresFile :
            tokens = line.split()
            scores[int(tokens[0]) - 1] = float(tokens[2])
            pvals[int(tokens[0]) - 1] = float(tokens[3])

        self.pvals = pvals
        return scores

    def score2(self, data, **args) :

        featuresPerForest = int(math.ceil(float(self.maxSize) / len(data))) - 10
        numForests = int(math.ceil(float(data.numFeatures) /
                                   featuresPerForest))
        perm = range(data.numFeatures)
        random.shuffle(perm)
        scores = numpy.zeros(data.numFeatures, numpy.Float)
        pvals = numpy.zeros(data.numFeatures, numpy.Float)
        featureIDdict = misc.list2dict(data.featureID, range(data.numFeatures))
        print 'numForests', numForests, featuresPerForest, data.numFeatures
        for i in range(numForests) :
            print 'forest number', i + 1
            if i < numForests - 1 :
                features = perm[featuresPerForest * i :
                                featuresPerForest * (i + 1)]
            else :
                features = perm[featuresPerForest * i : ]
            subdata = data.__class__(data, 'deepcopy')
            subdata.keepFeatures(features)
            subscores = self.score(subdata, **args)
            for j in range(subdata.numFeatures) :
                scores[featureIDdict[subdata.featureID[j]]] = subscores[j]
                pvals[featureIDdict[subdata.featureID[j]]] = self.pvals[j]
        # re-rank all the best features together:
        bestFeatures = numpy.argsort(scores)[:featuresPerForest]
        print featuresPerForest
        print 'length of best Features', len(bestFeatures)
        subdata = data.__class__(data, 'deepcopy')
        subdata.keepFeatures(bestFeatures)
        subscores = self.score(subdata, **args)
        for j in range(subdata.numFeatures) :
            scores[featureIDdict[subdata.featureID[j]]] = subscores[j]
            pvals[featureIDdict[subdata.featureID[j]]] = self.pvals[j]

        self.pvals = pvals
        return scores
    
    def train(self, data, **args) :

        Classifier.train(self, data, **args)
        self.featureID = data.featureID[:]
        
        if data.numFeatures * len(data) > 1.1 * self.maxSize :
            self.train2(data, **args)
        
        self.cleanup()
        # the location of the output from training:
        #self.trainingDirectory = tempfile.mkdtemp()
        self.trainingDirectory = '/Users/asa/temp'
        print 'RF directory:', self.trainingDirectory
        self.trainingExecutable = os.path.join(self.trainingDirectory, 'rf')
        self.trainingCode = os.path.join(self.trainingDirectory, 'rf.f')
        #os.mkdir(self.trainingDirectory)
        writeData(data)
        writeCode(self, self.trainingCode, self.trainingExecutable, data)
        dir = os.getcwd()
        os.chdir(self.trainingDirectory)
        os.system('./rf')
        os.chdir(dir)
        
    def test(self, data, **args) :

        testStart = time.clock()

        patterns = range(len(data))
        self.testingExecutable = os.path.join(self.trainingDirectory, 'rft')
        self.testingCode = os.path.join(self.trainingDirectory, 'rft.f')
        writeData(data, patterns, '/tmp/data.test', state='testing')
        writeCode(self, self.testingCode, self.testingExecutable,
                  data, len(patterns), state='testing')
        dir = os.getcwd()
        os.chdir(self.trainingDirectory)
        # remove some files so the fortran code won't complain
        os.system('rm -f savedparams')
        os.system('rm -f save-*')
        os.system('./rft')

        # read the output produced by RF:
        file = open('save-data-from-run')

        res = self.resultsObject(data, self, **args)

        i = 0
        for line in file :
            tokens = line.split()
            y = int(tokens[2]) - 1
            if self.labels.numClasses == 2 :
                decisionFunc = float(tokens[4]) - float(tokens[3])
            else :
                decisionFunc = 0
            res.appendPrediction((y, decisionFunc), data, patterns[i])
            i += 1
        os.chdir(dir)

        try :
            res[0].log = classifier.log
        except :
            pass
        try :
            computeStats = args['stats']
        except :
            computeStats = False
        if computeStats and data.labels.L is not None :
            res.computeStats()

        try :
            res[0].log.testingTime = time.clock() - testStart
        except :
            pass

        return res

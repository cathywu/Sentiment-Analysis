import os
import random
import numpy

from PyML.utils import misc
from PyML.classifiers.baseClassifiers import Classifier
from PyML.classifiers.ext.libsvm import C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR
from PyML.classifiers.ext.libsvm import LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED
from PyML.classifiers.ext import csvmodel,libsvm,mylibsvm
from PyML.classifiers.ext import csvmodel
from PyML.classifiers.liblinear import mylinear

from PyML.utils import arrayWrap
from PyML.evaluators import assess,resultsObjects
from PyML.containers.vectorDatasets import VectorDataSet, SparseDataSet
from PyML.containers import ker

from PyML.classifiers.ext import csmo
from PyML.classifiers.ext import cgist


"""various flavors of SVMs and training algorithms"""

__docformat__ = "restructuredtext en"

containersNotSupported = ['PySparseDataSet', 'PyVectorDataSet']

class SVM (Classifier) :
    """
    An SVM classifier class.

    SVM is trained using either libsvm, or using a PyML SMO implementation
    based on libsvm
    """

    svm_type = C_SVC
    attributes = {'C' : 10,
                  'nu' : 0.5,
                  'Cmode': 'classProb',
                  'optimizer' : 'libsvm',
                  'cacheSize' : 256,
                  'nu' : 0.1,
                  'eps' : 0.01,
                  'loss' : 'l1'}
    
    def __init__(self, arg = None, **args):

        """
        :Parameters:
          - `arg` - another SVM object or a kernel object; if no argument is given
            the kernel function of the training dataset is used
        
        :Keywords:
          - `C` - the svm C parameter
          - `Cmode` - the way the C parameter is used; values: 'equal', 'classProb',
            'fromData'.
            In 'equal' mode C is set to be the same for both classes
            In 'classProb' mode each class is assigned a C value that is 
            proportional to the size of the other class.  This results in 
            margin error costs being proportional to the ratio of the
            sizes of the two classes.  
            This is useful for datasets with an unbalanced class distribution.
            In 'fromData' the value of C for each pattern is taken from the
            'C' attribute of the training data.
          - `optimizer` - which optimizer to use.  the options are:
            'libsvm' -- run libsvm
            'liblinear' -- use liblinear (linear svm only)
            in this case you have the option to set the loss function - either 'l1' or 'l2'
            'mysmo' - use the PyML native optmizer (based on libsvm)
            'gist' - use a gist-like optimizer.
          - `loss` - when using liblinear set this to 'l1' or 'l2' (default: 'l1')
          - `cacheSize` - size of the kernel cache (in MB).
        """

        Classifier.__init__(self, arg, **args)

        self.kernel = None
        if arg.__class__ == self.__class__ :
            if arg.kernel is not None :
                self.kernel = arg.kernel.__class__(arg.kernel)
        elif hasattr(arg, 'type') and arg.type == 'kernel' :
            self.kernel = arg.__class__(arg)
        elif arg is not None :
            raise ValueError, 'unknown type of argument'
            
    def __repr__(self) :

        rep = ['<' + self.__class__.__name__ + ' instance>']
        if hasattr(self, 'C') :
            rep.append('C : %f' % self.C)
            rep.append('Cmode: %s' % self.Cmode)
        if hasattr(self, 'kernel') and self.kernel is not None :
            rep.append(str(self.kernel))
        if hasattr(self, 'model') :
            if hasattr(self, 'model') :
                rep.append(str(self.model))
        
        return '\n'.join(rep)

    def save(self, fileName) :

        """
        save an SVM model to a file.
        use the loadSVM method to then load the saved model
	
        :Parameters:
          - `fileName` - a file name or file handle
        """

        self.model.save(fileName)
            
    def load(self, fileName, data) :

        if type(fileName) == type('') :
            file_handle = open(fileName)
        else :
            file_handle = fileName
        line = file_handle.readline()
        tokens = line[:-1].split(',')
        for token in tokens :
            if token.find('b=') >= 0 :
                b = float(token.split('=')[1])
            elif token.find('alpha=') >= 0 :
                alpha = [float(t) for t in token.split('=')[1].split()]
            elif token.find('svID=') >= 0 :
                svID = [int(t) for t in token.split('=')[1].split()]
        self.train(data, alpha = alpha, b = b, svID = svID)


    def train(self, data, **args) :

        """
        train an SVM
        """
    
        if data.__class__.__name__ in containersNotSupported :
            raise ValueError, 'convert your data into one of the C++ containers'

        Classifier.train(self, data, **args)
        if self.kernel is not None :
            data.attachKernel(self.kernel)

        # libsvm optimizer can only be used with vector data:
        if (not data.isVector) and self.optimizer == 'libsvm' :
            self.optimizer = 'mysmo'

        isPrimal = False
        if 'alpha' in args :
            print 'loading model'
            alpha = args['alpha']
            b = args['b']
            svID = args['svID']
        elif self.optimizer == 'libsvm' :
            alpha,b,svID = self.trainLibsvm(data, **args)
        elif self.optimizer == 'liblinear' :
            isPrimal = True
            w, b = self.trainLiblinear(data, **args)
        elif self.optimizer == 'gist' :
            alpha,b,svID = self.trainGist(data, **args)
        elif self.optimizer == 'gradient' :
            alpha,b,svID = self.trainGradient(data, **args)            
        else :
            alpha,b,svID = self.trainMySMO(data, **args)

        if isPrimal :
            self.model = self.modelDispatcher(data, w=w, b=b)
        else :
            self.model = self.modelDispatcher(data, svID=svID, alpha=alpha, b=b)

        self.trained = True
        if not isPrimal :
            self.log.numSV = len(alpha)
        self.log.trainingTime = self.getTrainingTime()


    def modelDispatcher(self, data, **args) :

        if ('w' in args or (data.kernel.__class__.__name__.find('Linear') == 0
            and data.isVector) ) :
            return LinearSVModel(data, **args)
        else :
            return SVModel(data, **args)
        
    def trainLibsvm(self, data, **args) :
        
        # setting C for the positive and negative classes
        if (self.svm_type == ONE_CLASS or
            self.svm_type == EPSILON_SVR or
            self.svm_type == NU_SVR) :
            Cpos = 0
            Cneg = 0
        else :
            if data.labels.numClasses != 2 :
                raise ValueError, 'svm is a two class classifier'
            Cpos, Cneg = self.getC(data)
            
        print 'Cpos, Cneg: ', Cpos,Cneg

        # prepare data for the libsvm wrapper :
        # set kernel:
        if hasattr(self, 'kernel') and self.kernel is not None :
            kernel = self.kernel
        else :
            kernel = data.kernel
        kernelType = kernel.__class__.__name__

        param = libsvm.svm_parameter()
        misc.update(param, 
                    kernel_type = LINEAR,
                    svm_type = self.svm_type,
                    cache_size = self.cacheSize,
                    eps = self.eps,
                    C = self.C,
                    nu = self.nu,
                    degree = 2,
                    p = 0.1,
                    shrinking = 1,
                    nr_weight = 0,
                    coef0 = 0)

        if kernelType == "Polynomial" :
            # (gamma x' y + coef0)^degree
            param.kernel_type = POLY
            param.degree = kernel.degree
            param.coef0 = kernel.additiveConst
            param.gamma = 1
        elif kernelType == "Gaussian":
            # exp(-gamma * |x - y|^2)
            param.kernel_type = RBF
            param.gamma = kernel.gamma
        elif kernelType == "Cosine" :
            # i'm using the sigmoid kernel as the cosine kernel
            param.kernel_type = SIGMOID
            
        s=libsvm.DecisionFunction()

        prob = libsvm.svm_problem()
        data.libsvm_construct(prob)
        libsvm.svm_train_one_pyml(prob.this, param.this, Cpos, Cneg, s.this)
        mylibsvm.libsvm_destroy(prob)

        b = -s.rho

        numSV = s.numSV
        alpha = arrayWrap.doubleVector2list(s.alpha)
        svID = arrayWrap.intVector2list(s.svID)

        return alpha, b, svID

    def getC(self, data) :

        if self.Cmode == "classProb":
            Cpos = self.C * (float(data.labels.classSize[0]) / float(len(data)))
            Cneg = self.C * (float(data.labels.classSize[1]) / float(len(data)))
        else:
            Cpos, Cneg = self.C
        return Cpos, Cneg

    def getClist(self, data) :

        if self.Cmode == "fromData" :
            C = data.C
        elif self.Cmode == "classProb":
            Cpos = self.C * (float(data.labels.classSize[0]) / float(len(data)))
            Cneg = self.C * (float(data.labels.classSize[1]) / float(len(data)))
            c = [Cneg, Cpos]
            C = [c[data.labels.Y[i]] for i in range(len(data))]
        else:
            C = [self.C for i in range(len(data))]

        return C

    def trainGist(self, data, **args) :

        if data.labels.numClasses != 2 :
            raise ValueError, 'svm is a two class classifier'

        alpha, b = runGist(self, data)

        svID = [i for i in range(len(alpha))
                if alpha[i] > 0]
        alpha = [alpha[i] * (data.labels.Y[i] * 2 - 1) for i in range(len(alpha)) 
                 if alpha[i] > 0]

        return alpha, b, svID

    def trainGradient(self, data, **args) :

        if data.labels.numClasses != 2 :
            raise ValueError, 'svm is a two class classifier'

        alpha, b = runGradientDescent(self, data)

        svID = [i for i in range(len(alpha))
                if alpha[i] > 0]
        alpha = [alpha[i] * (data.labels.Y[i] * 2 - 1) for i in range(len(alpha)) 
                 if alpha[i] > 0]

        return alpha, b, svID
        
    
    def trainMySMO(self, data, **args) :

        if data.labels.numClasses != 2 :
            raise ValueError, 'svm is a two class classifier'
        print 'training using MySMO'
        alpha, b = runMySMO(self, data)
        svID = [i for i in range(len(alpha))
                if alpha[i] > 0]
        alpha = [alpha[i] * (data.labels.Y[i] * 2 - 1) for i in range(len(alpha)) 
                 if alpha[i] > 0]
        b = - b

        return alpha, b, svID

    def trainLiblinear(self, data, **args) :

        print 'training using liblinear'
        #data.addFeature('bias', [1.0 for i in range(len(data))])
        Cpos, Cneg = self.getC(data)
        if self.loss == 'l1' :
            solver = 1
        else :
            solver = 3
        print 'solver', str(solver)
        w = mylinear.solve_l2r_l1l2_svc(data, self.eps, Cpos, Cneg, solver)
        #data.eliminateFeatures(['bias'])
        return w, 0

    def decisionFunc(self, data, i) :

        return self.model.decisionFunc(data, i)

    def classify(self, data, i) :

        margin = self.decisionFunc(data, i)
        if margin > 0 :
            return (1,margin)
        else:
            return (0,margin)


def loadSVM(fileName, data) :
    """
    returns a trained SVM object constructed from a saved SVM model.
    You also need to provide the data on which the SVM was originally trained.
    """

    s = SVM()
    s.load(fileName, data)
    return s


class SVR (SVM) :
    """A class for SVM regression (libsvm wrapper).
    """

    svm_type = EPSILON_SVR
    resultsObject = resultsObjects.RegressionResults
    classify = SVM.decisionFunc
    stratifiedCV = assess.cv

    def __repr__(self) :
        rep = '<' + self.__class__.__name__ + ' instance>\n'

        return rep
    

class OneClassSVM (SVM) :
    """wrapper for the libsvm one-class SVM"""

    svm_type = ONE_CLASS
    resultsObject = misc.DecisionFuncResults
    
    def __repr__(self) :

        rep = '<' + self.__class__.__name__ + ' instance>\n'

        return rep
                 
class SVC (Classifier) :

    attributes = {'lineSampleSize' : 10,
                  'nu' : 0.1,
                  'eps' : 0.001}


    def __init__(self, arg=None, **args) :

        Classifier.__init__(self, arg, **args)

    def train(self, data, **args) :

        Classifier.train(self, data, **args)
        self.oneClass = OneClassSVM(nu = self.nu, eps = self.eps)
        self.oneClass.train(data)
        self.data = data
        print 'computing connected components'
        self.clusters = self.connectedComponents()

    def decisionFunc(self, data, i) :

        return self.oneClass.decisionFunc(data, i)

    def classify(self, data, i) :

        margin = self.decisionFunc(data, i)
        if margin > 0 :
            return (1,margin)
        else:
            return (0,margin)

        
    def adjacent(self, i, j) :

        xi = numpy.array(self.data.getPattern(i))
        xj = numpy.array(self.data.getPattern(j))
        stepSize = 1.0 / (self.lineSampleSize + 1)
        lambdas = numpy.arange(0, 1, stepSize)
        X = []
        for l in lambdas[1:] :
            X.append((xi * l + xj * (1 - l)).tolist())
        testdata = VectorDataSet(X)
        
        for i in range(len(testdata)) :
            f = self.decisionFunc(testdata, i)
            if f < 0 :
                return False
        return True

    def connectedComponents(self) :

        # the set of patterns that do not belong in a connected component
        patterns = set(range(len(self.data)))
        # start with an empty set of connected components (clusters):
        clusters = []
        # all the patterns that are currently in a cluster:
        incluster = set()
        while len(patterns) > 0 :
            cluster = set()
            fringe = [patterns.pop()]
            while fringe :
                pattern = fringe.pop()
                if pattern not in cluster :
                    cluster.add(pattern)
                    if pattern in patterns : patterns.remove(pattern)
                    incluster.add(pattern)
                    fringe.extend([neighbor for neighbor in patterns
                                   if self.adjacent(pattern, neighbor)])

            clusters.append([i for i in cluster])

        return clusters


class SVModel (object) :

    def __init__(self, data, svID, alpha, b, **args) :

        self.alpha = alpha
        self.b = b
        self.svID = svID
        self.numSV = len(svID)
        if not data.isWrapper :
            self.svdata = data.__class__(data, patterns = svID)
        if data.isWrapper :
            self.cmodel = csvmodel.SVModel(data.castToBase(), svID, alpha, b)

    def __repr__(self) :

        rep = '<' + self.__class__.__name__ + ' instance>\n'
        rep += 'number of SVs: %d\n' % len(self)

        return rep
    
    def __len__(self) :

        return self.numSV

    def setBias(self, bias) :

        self.b = bias
        if hasattr(self, 'cmodel') :
            self.cmodel.b = bias

    def decisionFunc(self, data, i) :

        if hasattr(self, 'cmodel') :
            return self.cmodel.decisionFunc(data.castToBase(), i)
        sum = 0.0
        for j in range(len(self)) :
            sum += self.svdata.kernel.eval(
                self.svdata, self.svdata.X[j], data.X[i]) * self.alpha[j]

        return sum + self.b
    
    def save(self, fileName) :

        if type(fileName) == type('') :
            outfile = open(fileName, 'w')
        else :
            outfile = fileName

        outfile.write('b=' + str(self.b) + ',')
        outfile.write('alpha=' + ' '.join([str(a) for a in self.alpha]) + ',')
        outfile.write('svID=' + ' '.join([str(s) for s in self.svID]) + '\n')


class LinearSVModel (SVModel) :

    def __init__(self, data, **args) :

        if 'alpha' in args :
            self.alpha = args['alpha']
            self.svID = args['svID']
            self.numSV = len(self.svID)
            self.dual = True
        if 'w' in args :
            self.w = args['w']
            self.dual = False
        self.b = 0.0
        if 'b' in args :
            self.b = args['b']

        if not data.isWrapper :
            self.svdata = data.__class__(data, patterns = svID)

        if self.dual :
            if data.__class__.__name__ == 'SparseDataSet' :
                self.cmodel = csvmodel.LinearSparseSVModel(data, self.svID, self.alpha, self.b)
            else :
                self.cmodel = csvmodel.LinearSVModel(data, self.svID, self.alpha, self.b)
            self.w = self.cmodel.getWvec();
            self.warray = self.w
        else :
            self.cmodel = csvmodel.LinearSparseSVModel(data, self.w, self.b)

    def __repr__(self) :

        rep = '<' + self.__class__.__name__ + ' instance>\n'
        if self.dual :
            rep += 'number of SVs: %d\n' % len(self)

        return rep

    def decisionFunc(self, data, i) :
        
        return self.cmodel.decisionFunc(data, i)


def runMySMO(svmInstance, data) :

    C = svmInstance.getClist(data)
    alpha = csmo.runSMO(data.castToBase(), C, int(svmInstance.cacheSize))

    return alpha[:-1],alpha[-1]

def runGist(classifier, data) :

    C = classifier.getClist(data)
    alphaVec = arrayWrap.doubleVector()
    cgist.runGist(data.castToBase(), C, alphaVec,
                  int(classifier.cacheSize), 10000)
    alpha = [alphaVec[i] for i in range(len(alphaVec))]

    return alpha, 0.0

def runGradientDescent(classifier, data) :

    C = classifier.getClist(data)
    alphaVec = arrayWrap.doubleVector()
    cgist.runGradientDescent(data.castToBase(), C, alphaVec,
                             int(classifier.cacheSize), 10000)
    alpha = [alphaVec[i] for i in range(len(alphaVec))]

    return alpha, 0.0


import numpy

from PyML.classifiers.baseClassifiers import Classifier


class RidgeRegression (Classifier) :
    """
    A kernel ridge regression classifier

    :Keywords:
      - `ridge` -- the ridge parameter [default: 1.0]
      - `kernel` -- a kernel object [default: Linear]

    """

    attributes = {'ridge' : 1.0,
		  'kernel' : None}

    def __init__(self, arg = None, **args) :
    
        Classifier.__init__(self, arg, **args)

    def __repr__(self) :
        
        rep = '<' + self.__class__.__name__ + ' instance>\n'
        rep += 'ridge: %f\n' % self.ridge
            
        return rep
                    
        
    def train(self, data, **args) :

        Classifier.train(self, data, **args)
        self.data = data
        if self.kernel is not None :
            self.data.attachKernel(self.kernel)
            
        Y = numpy.array(data.labels.Y)
        Y = Y * 2 - 1

        K = numpy.zeros((len(data), len(data)), numpy.float_)
        print 'getting kernel matrix'
        for i in range(len(data) - 1) :
            for j in range(i, len(data)) :
                K[i][j] = data.kernel.eval(data, i, j)
                K[j][i] = K[i][j]
        K = K + self.ridge * numpy.eye(len(data))
        print 'about to call numpy.linalg.inv'
        self.alpha = numpy.dot(Y, numpy.linalg.inv(K))

        self.log.trainingTime = self.getTrainingTime()
        print 'done training'
        
    classify = Classifier.twoClassClassify

    def decisionFunc(self, data, i) :

        f = 0.0
        for j in range(len(self.data)) :
            f += self.alpha[j] * self.data.kernel.eval(data, i, j, self.data)
            
        return f



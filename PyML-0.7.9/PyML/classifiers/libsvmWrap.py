from ext import libsvm

from ext.libsvm import svm_problem
from ext.libsvm import svm_parameter
from ext.libsvm import C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR
from ext.libsvm import LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED


class svmParameter (svm_parameter) :

    def __init__(self, **kw) :

        svm_parameter.__init__(self)
        
        self.svm_type = C_SVC
        self.kernel_type = LINEAR
        self.degree = 2
        self.gamma = 0
        self.coef0 = 0
        self.nu = 0.5
        self.cache_size = 40
        self.C = 10
        self.eps = 1e-3
        self.p = 0.1
        self.shrinking = 1
        self.nr_weight = 0
        for attr,val in kw.items():
            setattr(self,attr,val)


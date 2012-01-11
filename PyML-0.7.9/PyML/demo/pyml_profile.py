
"""
compare running times of several svm solvers
"""

from PyML import *

def read_yeast_data() :

    return SparseDataSet('yeast2.data', labelsColumn =1)

def read_rcv1_data() :

    return SparseDataSet('rcv1_train.binary')

def run_libsvm(C = 1) :

    #data = read_yeast_data()
    data = read_rcv1_data()
    SVM(C=C).stratifiedCV(data, seed = 1)

def run_liblinear_l2(C = 1) :

    #data = read_yeast_data()
    data = read_rcv1_data()
    SVM(optimizer = 'liblinear', loss = 'l2', C=C).stratifiedCV(data, seed = 1)

def run_liblinear_l1(C = 1) :

    #data = read_yeast_data()
    data = read_rcv1_data()    
    SVM(optimizer = 'liblinear', loss = 'l1', C=C).stratifiedCV(data, seed = 1)

def profile_all() :

    import cProfile

    cProfile.run('run_liblinear_l2()')
    cProfile.run('run_liblinear_l1()')
    cProfile.run('run_libsvm()')
    
if __name__ == '__main__' :
    
    profile_all()

#!/usr/bin/python

import classifier
import data
from numpy import *
import numpy.random

def kfold(k, classifier_type, dat):
    print "Splitting array..."
    mat = dat.asMatrix().T.copy()
    numpy.random.shuffle(mat)
    folds = array_split(mat.T, k, axis=1)
    success = 0
    count = 0

    for i in range(len(folds)):
        print "Running fold", i
        testdata = folds[i]
        classif = classifier_type(data.Data(hstack(folds[:i] + folds[i+1:])))
        s, c = successrate(classif, data.Data(testdata))
        print (float(s)/float(c))
        success += s
        count += c
        
    return 1 - (float(success)/float(count))

def successrate(classif, testdata):
    count = 0
    success = 0
    for col in testdata.asMatrix().T:
        count += 1
        if not col[-1] == classif.classify(col[:-1]):
            success += 1
    return (success, count)

if __name__ == "__main__":
    d = data.Data(data.DefDict((), {

                (1,2,3) : (1,),
                (3,3,1) : (0,),
                (1,2,3) : (1,),
                (1,4,3) : (1,),
                (1,2,4) : (1,),
                (1,2,1) : (1,),
                (1,2,6) : (1,),
                (1,4,5) : (0,),
                (1,5,3) : (1,),
                (1,6,3) : (0,)
                }))
    print d.asMatrix()
    
    print "-----------"
    print kfold(5, classif, d)    

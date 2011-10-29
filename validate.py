import classifier
import data
from numpy import *

def kfold(k, classifier_type, dat):
    folds = array_split(dat.asMatrix(), k, axis=1)
    error = 0
    count = 0

    for i in range(len(folds)):
        testdata = folds[i]
        classif = classifier_type(hstack(folds[:i] + folds[i+1:]))
        e, c = errorrate(classif, data.Data(testdata))
        error += e
        count += c
    return float(error)/float(count)

def errorrate(classif, testdata):
    count = 0
    error = 0
    for vec in testdata.asDict():
        count += 1
        for v in testdata.asDict()[vec]:
            if not v == classif.classify(vec):
                error += 1
    return (error, count)
        
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
    classif = classifier.OneClassifier
    print d.asMatrix()
    
    print "-----------"
    print kfold(5, classif, d)
    

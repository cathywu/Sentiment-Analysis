
import os

import datafunc
import svm
import preproc
import composite
import modelSelection
import ker

def rescale(Clist = [0.1, 1, 10, 100], data = None, **args) :

    chain = composite.Chain([preproc.Rescale(), svm.SVM(**args)])
    if Clist is not None :
        param = modelSelection.Param(chain, 'classifier.C', Clist)
        return modelSelection.ModelSelector(param)
    else :
        return chain

def rescaleGaussian(gammaList = [0.01, 0.05, 0.1, 0.3, 1, 2], data = None, **args) :

    k = ker.Gaussian()
    s=svm.SVM(k)
    chain = composite.Chain([preproc.Rescale(), s])
    if gammaList is not None :
        param = modelSelection.Param(chain,
                                     'classifier.kernel.gamma', gammaList)
        return modelSelection.ModelSelector(param)
    else :
        return chain

def gaussianSelect(gammaList = [0.01, 0.05, 0.1, 0.3, 1, 2], **args) :

    measure = 'balancedSuccessRate'
    if 'measure' in args :
	measure = args['measure']
    k = ker.Gaussian()
    param = modelSelection.Param(svm.SVM(k),
				 'kernel.gamma', gammaList)
    return modelSelection.ModelSelector(param, measure = measure)


def RF(data = None, **args) :

    from PyCode.PyML import randomForests2
    rf = randomForests2.RF()

    return rf

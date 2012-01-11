import os
import numpy

from PyML.classifiers import svm,multi,ridgeRegression,knn,composite,modelSelection
from PyML.classifiers.svm import SVM
from PyML.feature_selection import featsel
from PyML.containers import ker,labels
from PyML.containers import vectorDatasets
from PyML.containers.aggregate import Aggregate
from PyML.containers.kernelData import KernelData
from PyML.containers.sequenceData import SequenceData
from PyML.classifiers import platt
from PyML.preproc import preproc

from PyML.evaluators import resultsObjects

#heartdatafile = '../../data/heart.data'
heartdatafile = 'heart.data'
#irisdatafile = '../../data/iris.data'
irisdatafile = 'iris.data'
#yeastdatafile = '../../data/yeast.data'
yeastdatafile = 'yeast.data'

def test (component = 'svm', **args) :

    if 'container' in args :
        container = args['container']
    else :
        container = 'SparseDataSet'

    try :
        DataSet = getattr(vectorDatasets, container)
    except :
        raise ValueError, 'wrong container ' + container

    s = svm.SVM()

    results = {}

    comp = 'general'
    if component == 'all' or component == comp :
        d = DataSet (heartdatafile, labelsColumn = 0)
        s.train(d)
        s.test(d)
        s = svm.SVM()
        s.stratifiedCV(d)
        d2 = Aggregate([d,d])
        #r = s.stratifiedCV(d2)
        #r = s.loo(d)
        d.attachKernel('polynomial')
        s.cv(d)
        d.attachKernel('linear')
        s = svm.SVM()
        s.train(d)

        d = DataSet(numpy.random.randn(100,10))
        d = DataSet([[1,2], [2,3]])
        #d = SequenceData(['asa', 'ben', 'hur'])
        
    comp = 'svm'
    if component == 'all' or component == comp :
        d = DataSet (heartdatafile, labelsColumn = 0)
        results[comp] = []
        d.attachKernel('polynomial')
        results[comp].append(
            s.cv(d, saveSpace = True))
        d.attachKernel('linear')
        results[comp].append(
            s.cv(d))
        results[comp].append(SVM(optimizer = 'liblinear', loss = 'l1', C=1).stratifiedCV(d, seed = 1))
        results[comp].append(SVM(optimizer = 'liblinear', loss = 'l2', C=1).stratifiedCV(d, seed = 1))
        
    comp = 'kernelData'
    if component == 'all' or component == comp :
        d = DataSet (heartdatafile, labelsColumn = 0)
        results[comp] = []
        kdata = KernelData('heart.kernel', gistFormat = True)
        kdata.attachLabels(d.labels)
        s=svm.SVM()
        results[comp].append(
            s.cv(kdata))
        kdata.attachKernel('gaussian', gamma = 0.1)
        results[comp].append(
            s.cv(kdata))

    comp = 'normalization'
    if component == 'all' or component == comp :
        results[comp] = []
        data = DataSet (heartdatafile, labelsColumn = 0)
        data.attachKernel('polynomial', degree = 4, normalization = 'dices')
        s=svm.SVM()
        results[comp].append(
            s.cv(data))

    comp = 'svr'
    if component == 'all' or component == comp :
        d = DataSet (heartdatafile, labelsColumn = 0, numericLabels = True)
        results[comp] = []
        s = svm.SVR()
        #results[comp].append(
        #    s.cv(d, saveSpace = True))
        #results[comp].append(
        #    s.trainTest(d, range(150), range(151, 250)))
        results[comp].append( s.cv(d) )

    comp = 'save'
    if component == 'all' or component == comp :
        results[comp] = []
        s = svm.SVM()
        data = DataSet (heartdatafile, labelsColumn = 0)
        import tempfile
        tmpfile = tempfile.mktemp()
        r = s.cv(data)
        r.save(tmpfile)
        r = resultsObjects.loadResults(tmpfile)
        results['save'].append(r)


        r = s.nCV(data)
        r.save(tmpfile)
        results['save'].append(resultsObjects.loadResults(tmpfile))

        r = {}
        for i in range(10) :
            r[i] = s.cv(data)

        resultsObjects.saveResultObjects(r, tmpfile)
        r = resultsObjects.loadResults(tmpfile)
        
    comp = 'classifiers'
    if component == 'all' or component == comp :
        d = DataSet (heartdatafile, labelsColumn = 0)
        results[comp] = []
        cl = knn.KNN()
        results[comp].append(
            cl.stratifiedCV(d))
        print 'testing ridge regression'
        ridge = ridgeRegression.RidgeRegression()
        #results[comp].append(
        #    ridge.cv(d))

    comp = 'platt'
    if component == 'all' or component == comp :
        results[comp] = []
        d = DataSet (heartdatafile, labelsColumn = 0)
        p = platt.Platt2(s)
        results[comp].append(p.stratifiedCV(d))

    comp = 'save'
    if component == 'all' or component == comp :        
        data = DataSet (heartdatafile, labelsColumn = 0)
        s = SVM()
        s.train(data)
        s.save('model.pyml')
        s2 = SVM()
        s2.load('model.pyml', data)
        results[comp].append(s2.test(data))

        d = DataSet (heartdatafile, labelsColumn = 0)
        kdata = KernelData('heart.kernel', gistFormat = True)
        kdata.attachLabels(d.labels)
        s=svm.SVM()
        s.train(data)
        s.save('model2.pyml')
        s2 = SVM()
        s2.load('model2.pyml', kdata)
        results[comp].append(s2.test(kdata))

        data = DataSet(irisdatafile, labelsColumn = -1)
        mc = multi.OneAgainstRest(SVM())
        mc.train(data)
        mc.save('iris.pyml')
        mc = multi.OneAgainstRest(SVM())
        mc.load('iris.pyml', data)
        results[comp].append(mc.test(data))

    comp = 'multi'
    if component == 'all' or component == comp :
        results[comp] = []
        d = DataSet(irisdatafile, labelsColumn = -1)

        mc = multi.OneAgainstOne (svm.SVM())
        results[comp].append(
            mc.cv(d))

        d = DataSet(irisdatafile, labelsColumn = -1)
        
        mc = multi.OneAgainstRest (svm.SVM())
        results[comp].append(
            mc.cv(d))

        mc = multi.OneAgainstRest (svm.SVM())
        d.attachKernel('poly')
        results[comp].append(
            mc.cv(d))
        d.attachKernel('linear')
        mc = multi.OneAgainstRest (svm.SVM())
        #kdata = datafunc.KernelData('iris.linear.kernel',
        #                            labelsFile = 'irisY.csv', labelsColumn = 0, gistFormat = True)
        #results[comp].append(mc.cv(kdata))

        
    comp = 'featsel'
    if component == 'all' or component == comp :
        results[comp] = []
        
        s = svm.SVM()
        d = DataSet (yeastdatafile, labelsColumn = 0)
        d2 = labels.oneAgainstRest(d, '2')
        results[comp].append(
            s.stratifiedCV(d2, seed = 1))

        # feature selection
        m = composite.FeatureSelect (s, featsel.RFE())
        results[comp].append(
            m.stratifiedCV(d2, seed = 1))

        d = DataSet (yeastdatafile, labelsColumn = 0)
        d2 = labels.oneAgainstRest(d, '2')
        fs = featsel.FeatureScore ('golub')
        f = featsel.Filter (fs, sigma = 2)
        m = composite.FeatureSelect (s, f)
        results[comp].append(
            m.stratifiedCV(d2, seed = 1))

        # same thing but with a Chain:
        c = composite.Chain ([f,s]) 
        #r = c.stratifiedCV (d2)

    comp = 'modelSelection'
    if component == 'all' or component == comp :
        results[comp] = []
        s = svm.SVM()
        d = DataSet (heartdatafile, labelsColumn = 0)
        p = modelSelection.ParamGrid(svm.SVM(ker.Polynomial()), 'C', [0.1, 1, 10, 100],
                                     'kernel.degree', [2, 3, 4])
        p = modelSelection.ParamGrid(svm.SVM(ker.Gaussian()), 'C', [0.1, 1, 10, 100],
                                     'kernel.gamma', [0.01, 0.1, 1])
        #p = modelSelection.Param(svm.SVM(), 'C', [0.1, 1, 10, 100])

        m = modelSelection.ModelSelector(p, measure = 'roc', foldsToPerform = 2)
        m = modelSelection.ModelSelector(p)
        #m = modelSelection.SVMselect()
        results[comp].append(
            m.cv(d))

    comp = 'preproc'
    if component == 'all' or component == comp :
        results[comp] = []
        
        s = svm.SVM()
        d = DataSet (yeastdatafile, labelsColumn = 0)
        d2 = labels.oneAgainstRest(d, '2')
        results[comp].append(
            s.stratifiedCV(d2))
        p = preproc.Standardizer()
        p.train(d2)
        results[comp].append(
            s.stratifiedCV(d2))
        print p.scale
        print p.translation

    return results

if __name__ == '__main__' :

    if len(sys.argv) > 1 :
        test(sys.argv[1])
    else :
        test()


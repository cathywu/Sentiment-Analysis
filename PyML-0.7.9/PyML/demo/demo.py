    
import datafunc
import svm
import multi
import assess
import myio
import ker
import classifiers
import composite
import featsel
import modelSelection

# read data
d = datafunc.SparseCDataSet ('heartSparse.data')

# look at the data
print d

# construct svm classifier
s = svm.SVM()

# train
s.train(d)

# test by CV
r=s.cv(d)

# look at the results:
print r

# show ROC curve
r.plotROC()

# save results
r.save('test.pyd')
r=myio.load('test.pyd')

# try polynomial kernel
d.attachKernel('polynomial')
rp=s.cv(d)

# another way of doing this:
d.attachKernel('linear')
s = svm.SVM(ker.Polynomial())
r=s.cv(d)

d.attachKernel('linear')
knn = classifiers.KNNC()
r=knn.cv(d)

d = datafunc.DataSet ('heartSparse.data')
ridge = classifiers.RidgeRegression()
r=ridge.cv(d)


d = datafunc.SparseCDataSet('iris.data', labelsColumn = -1)
mc = multi.OneAgainstOne (svm.SVM())
r=mc.cv(d)

mc = multi.OneAgainstRest (svm.SVM())
r=mc.cv(d)
        
s = svm.SVM()
d = datafunc.SparseDataSet ('yeast.data', labelsColumn = 0)
d = datafunc.oneAgainstRest(d, '2')
r=s.cv(d)

m = composite.FeatureSelect (s, featsel.RFE())
r=m.cv(d, 3)
        
fs = featsel.FeatureScore ('golub')
f = featsel.Filter (fs, sigma = 2)
m = composite.FeatureSelect (s, f)
r=m.cv(d,3)

d = datafunc.SparseDataSet ('heart.data')
p = modelSelection.Param(svm.SVM(), 'C', [0.1, 1, 10, 100, 1000])
m = modelSelection.ModelSelector(p)
m.train(d)


d = datafunc.SparseDataSet ('heartSparse.data')
p = modelSelection.Param(classifiers.KNN(), 'k', [1,2,3,5,10,15])
m = modelSelection.ModelSelector(p)
m.train(d)



r = p.cv(d, numFolds = 10)
results = [r for r in p.cv(d, numFolds = 10)]
results = [r.successRate for r in p.cv(d, numFolds = 10)]

d = datafunc.SparseDataSet ('yeast.data', labelsColumn = 0)

d = datafunc.SparseDataSet ('yeast2.data', labelsColumn = 1)



from PyML import *

d = datafunc.VectorDataSet ('yeast3.data', labelsColumn = 1)

knn = classifiers.KNN()
results = knn.stratifiedCV(d)

p = modelSelection.Param(classifiers.KNN(), 'k', [1,2,3,5,10,15])

results = [r.successRate for r in p.stratifiedCV(d)]

m = modelSelection.ModelSelector(p)








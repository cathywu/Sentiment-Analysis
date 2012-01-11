
from PyML import *
data = VectorDataSet ('yeast.data', labelsColumn = 0)
data = SparseDataSet ('yeast.data', labelsColumn = 0)
data = labels.oneAgainstRest(data, '2')

f = featsel.RFE()
f.train(data)
print data.numFeatures, data.featureID

from PyML import *
data = VectorDataSet ('yeast.data', labelsColumn = 0)
data = labels.oneAgainstRest(data, '2')

fs = featsel.FeatureScore ('golub')
f = featsel.Filter (fs, sigma = 2)
f.train(data)
print data.numFeatures, data.featureID


from PyML import *
data = SparseDataSet ('yeast.data', labelsColumn = 0)
data = VectorDataSet ('yeast.data', labelsColumn = 0)
data = labels.oneAgainstRest(data, '2')
m = composite.FeatureSelect(SVM(), featsel.RFE())
results = m.stratifiedCV(data)

# You can examine the features that were selected in each fold by looking at the
# training log.  For example, the features that were used in the first fold are
# extracted by:
print results.getLog()[0]

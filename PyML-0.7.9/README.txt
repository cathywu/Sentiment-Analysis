PyML - a Python Machine Learning package

Asa Ben-Hur
asa@cs.colostate.edu

PyML is a flexible Python framework for using various classification 
methods including Support Vector Machines (SVM).

Installation

Only Unix/Linux/OS-X is supported.
Requirements:  
The numpy package http://sourceforge.net/projects/numpy

A setup.py script is provided so installation follows the standard python idiom:

>>> python setup.py build
>>> python setup.py install

See the Installation section in the tutorial for a trouble-shooting guide.

Documentation:

Tutorial - see doc/tutorial.pdf 
Module documentation - see doc/autodoc/public/index.html
Usage examples - see PyML/demo/pyml_test.py

PyML 0.7.9

- Fixed compilation issue under gcc 4.6.1 (tracker ID: 3361193)
- Fixed issue with VectorDataSet (tracker ID: 3364921)
- Updated tutorial

PyML 0.7.8
- Added wrapper for liblinear linear SVMs.  If you only need a linear
  SVM, these solvers offer a very significant speedup for large
  datasets.
  Usage:
     SVM(optimizer = 'liblinear', loss = 'l2') for l2 loss SVM
  or
     SVM(optimizer = 'liblinear', loss = 'l1') for l1 loss SVM
- Added containers.setData.SetData - a dataset container where each
example is a set of objects.
- Chris Hiszpanski reported an issue and fix for ROC calculation that
would fail for a corner case.
- When creating a dataset with numeric labels, the 'numericLabels'
keyword argument was not passed on to the Labels constructor when
creating such a dataset from arrays/lists
- the "stratifiedCV" method of SVR was being called by model
selection. That was addressed by defining it to be regular
cross-validation (stratified CV doesn't make sense for regression).
Better solution would be to have a separate base class for regression.
- Can create an empty SparseDataSet or VectorDataSet, and then
populate it on the fly with features (using its addFeatures method).

PyML 0.7.7
- Fixed the classify method of the KNN classifier
  (cross-validation worked fine).
- Fixed issue with the __repr__ of the Results objects that was giving
  an error when results of an unlabeled dataset were being displayed.

PyML 0.7.6
- Added positional kmer dataset creation (an implementation of
  Sonenburg et al's weighted degree kernel) that represents the
  features explicitly.

PyML 0.7.5
- Reworked the SequenceData container
- Fixed a bug in Labels.oneAgainstRest (thanks to Marcel Luethi for
  bug report)
- Fixed a bug in feature selection - feature IDs are now correctly retained

PyML 0.7.4.2
- Fixed bug in SparseDataSet's addFeatures method

PyML 0.7.4.1
- Removed dependency on scipy

PyML 0.7.4
- Modified the way svm models are saved/loaded.  Now you can save
  any SVM model.  Note that the interface for loading/saving has
  changed.
  OneAgainstRest classifiers also support save/load.
- Fixed the import error from 0.7.3

PyML 0.7.3

- Fixed bug  2971509 reported by Vebjorn Ljosa ( ljosa ) 
- Updated the requirements for PyML (python 2.5 and up).
- Increased the flexibility of the Aggregate container
- ROC curve plotting:  when doing cross-validation, PyML 
  now plots an ROC curve produced by averaging ROC curves
  of individual folds.
- AUC calculation gives the correct result when the discriminant
  scores are concentrated on a few values.
- the assess module has been split into several modules under
  the evaluators directory

PyML 0.7.2

- Fixed bug in Aggregate container (in the case of a weighted 
  combination of datasets)  Bug report by Eithon Cadag
- Stephen Picolo found issue when using feature selection with the VectorDataSet 
  container.  Use SparseDataSet instead.

PyML 0.7.1

- added k-means clustering (PyML.clusterers.kmeans)
- corrected handling of nan's
- fixed an import statement in classifiers/modelSelect.py
- Compilation error in Kernel.h fixed (shows in gcc 4.3)

PyML 0.7.0.1

- preproc.pca - updated code to new version of numpy
- demo2d.scatter - improved interface to make it more useful

PyML 0.7.0

- a restructuring of the module structure.  see tutorial for details
- small changes in demo2d
- linear kernel wasn't normalizing properly
- myio.myopen now handles bz2 files as well
- myio.myopen does not open in universal newline support mode
  by default - it throws the pyml parser off
- improved the way SequenceData reads fasta files -- it can now
  extract labels using a user-supplied function that extracts
  the id and label out of the fasta header.
- added a method for generating spectrum kernels 
  (containers.sequenceData.spectrumData)

PyML 0.6.15

- the svm cacheSize for libsvm wasn't being set up correctly.
  thanks to David Doukhan for the bugfix.
- added an 'addFeatures' method to the SparseDataSet and VectorDataSet 
  containers
- cleaned up the VectorDataSet container
- a dataset's attachLabels now takes either a labels object or a file name.
- when saving a dataset, the format (csv or sparse) is chosen according to
  the type of the container (sparse containers save to sparse format and
  non-sparse containers save to csv).
- improved the discussion of model selection in the tutorial
- more options in demo2d and a change of format

PyML 0.6.14

- Fixed bug in KNN.cpp where in some cases the decision function value
  is sometimes wrong (thanks to Martial Hue the bug report and a fix).
- Fixed a bug in loading non-linear models (thanks to shiela reynolds 
  for the bug report).
- construction from an array is now in the form  
  datafunc.VectorDataset(x) instead of datafunc.VectorDataset(X=x)
- small changes in demo2d (with the help of Adam Labadorf)

PyML 0.6.13

- Updates to SequenceData, including the addition of the positional-kmer-kernel
  and a save method.
- Datasets now maintain the norms of the examples, so that computation of 
  cosine and Gaussian kernels is much faster.
- a 'sample' module has been added with function for creating samples from a
  dataset (with and without replacement).
- fixed a segfault in the Aggregate container
- streamlined SWIG wrapper files

PyML 0.6.12.1

- added __init__.py in the ext directory

PyML 0.6.12

- PyML now distributed under LGPL license.
- The list of labels input argument for the Labels constructor can now be
  any list-like object.
- Fixed bug in loading a saved ResultsList.
- removed usage of numpy.nonzero in assess.py
- debugged svm.save/loadSVM 
- debugged demo2d

PyML 0.6.11

- wrappers now use swig 1.3.31
- added release number accessible as PyML.__version__
- fixed a bug in datafunc.save
- cleaned up some of the functions in the datafunc module
- when performing CV it could happen that a class would not be represented
  in one of the folds, creating a cryptic error/segfault.  now this is 
  explicitly checked.

PyML 0.6.10.1

- Corrected compilation bug that only showed up in new gcc compilers (version 4.1.1).
  (Bug reported by Michael Hamilton).

PyML 0.6.10

- Compatibility with latest numpy 1.0 (array codes, numpy.nonzero)
- Rewrote the constructor of 'datafunc.Labels'.
- The 'translate' method of VectorDataSet went missing, and has been restored.
- Cleaned up the libsvm wrapper.
- Dealt with bug in the C++ dataset containers introduced in 0.6.9

PyML 0.6.9

- added VectorDataSet -- a C++ non-sparse vector container.
  this resulted in some renaming of the dataset containers.
- updated the libsvm wrapper -- now wrapping version 2.82.
  to simplify the update of the wrapper, the pure python datasets are 
  no longer supported for training svm classifiers (it's easy to convert!)
- removed references to Numeric from myio.py
- better handling of nan's as a result of prediction
- a few more tweaks of the RegressionResults class
- reworked demo2d to work with version 0.87 of matplotlib
- the check method of the sparse parser didn't handle correctly files
  with only a single line (thanks to Jian-Qiu for the pointer and fix).

PyML 0.6.8

- reworked the Results containers, including improved support for regression 
  (RegressionResults container)
- corrected bug in reading in data with numeric labels (the numericLabels flag wasn't working)
- corrected compilation issue in KNN.h (showed up under fedora core 5)
- changed the directory structure of the package; setup.py was updated accordingly.
- corrected bug in RFE that affected cases where the number of desired features
was greater than the number of features in the dataset.
- corrected a couple of import issues related to the transition from Numeric to numpy

PyML 0.6.7

- numpy is now used instead of Numeric
- featureCount/featureCounts for the python containers
- rewrote RegressionResults (thanks to Jian Qiu for his contribution).
  still some work to be done to make it really useful
- datasets can be read from gzipped files.
- took care of some compilation warnings.
- assess.significance now also reports the medians of the statistics.
- KNN now accepts a generic dataset container.
- there is no longer a need to explicitly set an optimizer for non-vector
  datasets.
- cv and stratified CV can now do partial CV.

PyML 0.6.6

- cleaned up copy constructor of KernelData, and unified it with the standard
  copy constructor.
- fixed bug in using libsvm solver with 'Cosine' kernel.
- assess.loadResults2 / assess.saveResultsObjects now handle lists of lists of 
  objects, or dictionaries of lists of results objects
- fixed bug in getKernelMatrix; the last element of the diagonal was missing.
- fixed bug in assess.loo
- ROC score is not shown for multi class problems.
- Platt object now tries a couple of times before it raises an exception
- ker.sortKernel2 -- sorts a kernel matrix according to a given list of patterns
- when non-normalized roc curve is computed the normalized area is returned.

PyML 0.6.5

- added multi.allOneAgainstRest for performing all one-against-rest classification
- made sure that when making a dataset from an array, that it's a Numeric array
- revamping of the ridge regression classifier (it's still pure python)
- numFolds in cv and stratifiedCV can also be set using a keyword parameter
- added a baseClassifiers.IteratorClassifier from which modelSelection.Param and
  composite.FeatureSelectAll inherit
- added FeatureSelectAll -- for a backward/forward selection method, this looks at
  accuracy as a function of the number of features.
- corrected bug in setting of k in KNN
- corrected bug in Gaussian kernel for the pure python datasets
- demo2d.py -- a module for playing with data in two dimensions


PyML 0.6.4

- more options in Platt classifiers (see documentation for details)
- added a 'split' method to the datafunc module; it randomly splits a dataset into two parts
- corrected bug in nCV (setting of number of iterations; this is now a keyword argument)
- Better parsing of pairData input files
- Corrected refcount initialization in KernelData

PyML 0.6.3.1

- Corrected bug in CSVparser.readLabels
- Corrected bug in Results object in a case of one fold
- Merged the KNN and KNNC classes so that the user doesn't
  need to worry about that.

PyML 0.6.3

- Corrected bugs in CSV parser
  autodetect of a header row has proven problematic.
  if your CSV file has a header row you now need to specify that 
  as a keyword argument (headerRow = True).  the header row allows you
  to specify feature IDs (think of it as an excel spreadsheet).
- corrected bug in construction of a dataset from a Numeric array

PyML 0.6.2

- More cleaning up of the Results object
- replaced matlab with pylab
- added a getKernel method to the DataSet interface
- added a ker.showKernel function that displays a kernel matrix
- corrected bug in saving results object (not enough attributes were saved)

PyML 0.6.1

- Added save method for Platt classifier
- PyML no longer works with python 2.2 (sub-classing of the 'list' object
  has a problem in that version of python).
- fixed problem in Aggregate -- it broke down when the datasets from which
  it was constructed went out of scope.  other improvements to this class.
- fixed bug for reconstructing saved nonlinear SVM.
- Saved SVM had a problem with unlabeled data since it didn't reload its
  classLabels attribute -- bug fixed.
- Results object has been re-designed.
  The Results object IS a list of Results_
  objects that each provide the results on a different part of the
  data as is the case when doing CV.
  ROC scores computed using CV were to computed in previous versions by 
  pooling the decision function values and computing an pooled ROC curve.
  This has been changed to an average ROC score that is the average
  of the ROC areas from the different folds.
- Results object has a 'toFile' method for saving to an ascii file
  rather than pickling the object
- Added a ResultsList object (used for combining CV results on the
  same dataset).  nCV is now a classifier method and returns a
  ResultsList object rather than a list of Results objects
- Gnuplot related functions have been removed.  If you use them, just
  get them from from an older version.
- the 'xml' option in saving a results object has been discontinued.
- confusionMatrix attribute in a Results object is now a list rather than
  a Numeric array for safety in pickling
- the sparse parser has a 'sparsify' keyword argument that is "False"
  by default.  "sparsify" means not add features whose value in the file
  is 0.
- improved the way ModelSelector recognizes selection via roc scores
- composite.Platt2 - a classifier for rescaling discriminant values into 
  probability estimates based on Platt's method, but with better numerical
  stability (contributed by Tobias Mann).

PyML 0.6.0

- addFeature method for the sparse dataset containers
- registered attributes: the user can register attributes to a container;
  those attributes are then copied on copy construction
- eliminated the "aux" attribute from a dataset -- no need for that now
  that registeredAttributes are available
- the interface for "testing" of a dataset has changed:  the input for
  the test function includes the training data.  The "test" function
  of the classifier calls the "test" function of the dataset; therefore 
  the training data is now saved in the classifier
- corrected error in setup.py in definition of pairdataset extension
- "save" method of SVM objects now working.  Thanks to Olav Zimmerman
  for his input.
- timing information for training and testing a classfier are saved in the
  Result object's log or cvLog attributes
- Corrected bug in parsing space delimited files
- reading files in gist format (including label files)
- added datafunc.sample for sampling a dataset
- CV/stratifiedCV can save intermediate results
- the Labels constructor and the dataset constructors take a keyword
  argument 'positiveClass' that tells the Labels constructor which of
  the class labels is the positive class.  If the class labels are 
  +1 and -1 or 1 and -1, the positive class is detected automatically.
- assess.resultsByFold -- Given a Results object obtained by CV, divide 
  it into a list of Results objects for each fold.
- PyML is now supposed to work with 64-bit machines:
  the problem was that the hash values generated by python are 64-bit
  numbers.  These hash values are used as feature IDs in the SparseCDataSet,
  so those had to be changed into "long" instead of "int".  Using a "myint"
  proved problematic since arbitrary templates for vectors are not yet
  supported in swig.
- Plugged a couple of memory leaks

PyML 0.5.4

- corrected bug in roc50 calculation
- corrected bugs related to SVR
- gist and gradient descent solvers for svm formulation without a bias
  term
- Results object has an attribute foldStart so that the patterns between
  in the range [foldStart[i]:foldStart[i+1]] were in cross-validation fold
  number i where i is in range(0,numFolds)
- Construction of a dataset from a Numeric array is now possible
- Improvements in assess.plotROC and assess.plotROCs functions

PyML 0.5.3

- Param and ParamGrid are iterators:
  Param.cv, Param.stratifiedCV, Param.loo etc return an iterator:
  p = Parm(...)
  for result in p.cv(...) :
      ...
- datafunc.BaseDataSet renamed datafunc.VectorDataSet, which is what it
  is.  datafunc.BaseDataSet is now a more generic base class
- a dataset container has a train method that calls the dataset's
  'trainingFunc' method with the keyword arguments it receives.
  a classifier first trains the data before training itself. 
  (BaseClassifier calls the data's train method).
  naturally, there is also a "test" method.
  assess.test calls a dataset's test method with the keyword arguments it
  received.  if you redefine a classifier's "test" methos be sure to call 
  the dataset's test method.
- a dataset's train/test/cv methods are called as: self.method(self, **args)
  rather than self.method(self, *options, **args)

PyML 0.5.2

- assess.cvFromFolds -- perform cv when training and testing examples for
  each fold are given
- assess.stratifiedCV now can receive a list of patterns that are to serve
  as training examples in all CV folds
- corrected bug in cosine kernel
- corrected bug in "normalize" function in the case of zero vectors.
- ker.combineKernels -- add or multiply kernels (instead of ker.addKernels)
- ker.sortKernel plus a few other more esoteric functions for dealing with
  kernel matrices
- patternID created by default for a dataset
- modelSelector object now also supports selection by area under the roc/rocN
  curve
- aggregate data object -- combines several c++ data objects into a single
  dataset; its dot product is the sum of the dot products of the individual
  dataset objects
- corrected bug in assess.scatter
- corrected bugs reported by users in ParamGrid and csv parser

PyML 0.5.1

- assess.plotROCs -- plotting of multiple ROC curves for easy comparison
- corrected bugs in PairData
- better save method for vector-based containers
- corrected bug in getPattern method of SparseCDataSet
- datafunc -- labels object remembers what classes it was made from
  in the case of copy construction.


PyML 0.5

C++ part received a major upgrade:
- new level of abstraction for dataset objects -- generic interface for dataset
  and kernel objects allows "plug and play" data and kernel objects.
  These work with the PyML SMO object for SVM training (code based on libsvm).
- Cleaner wrapping of the C++ dataset container
- KernelData C++ container for an in-memory kernel matrix.
  The dotProduct function returns the appropriate entry of the kernel matrix.
  Like all containers, a kernel function can be defined on top of the dotProduct.

minor changes:
- Results object can be constructed from a Container object
- data object supports an "aux" attribute which is a list which stores auxiliary
  information on each pattern.
  uses -- a value of "C" for svm training.
- write kernel to a file, now directly in C++ (kernelWrap.py)
- adding kernels that are stored in a file (kernelWrap.py)
- Data container for data whose patterns are composed of pairs of objects

- removed some "warts" from the code with the help of pychecker
- updated documentation

unreleased code:
Random Forests

PyML 0.43 (unreleased)

- preprocessing object 
- save method for svm classifiers (not really working yet)
- improved save method for dataset containers
- corrected bug in ModelSelector object (CV did not work)
- In some cases a decision function may return a 'nan'.  Python has
  some peculiarities in handling nan.  This situation is now handled.
- improved CSV parser; interface modified (indexing starts at 0).
- succ feature score
- Results object computes rocN scores where N is 50 by default

PyML 0.42

support for One-class SVM and SVM regression
speedup of reading csv files
memory save mode for linear svm, some memory saving in general and plugging
a couple of memory leaks
memory still leaks from the swig interface -- help would be apprecited.

PyML 0.41:

Correction of bugs introduced in 0.4

PyML 0.4:

New Features:

Addition of a sparse dataset implemented in C++ - PyML runs several times 
faster in some cases.
Use the KNNC nearest neighbor classifier with this container.

RFE and Filter methods do automatic selection of the number of features.

Multiplicative update (a.k.a zero-norm) feature selection.

Changes in the interfaces:

All the dataset classes now have a Kernel automatically attached to them, so
a dataset is actually a feature-space.

There is now one SVM class - the object knows what type of SVM (linear, 
nonlinear) to train according to the kernel attached to the data object on 
which it is trained.


Changes in the classifier interface:

The "test" method of a classifier receives a dataset and a list of patterns
on which to test.

The "classify" and "decisionFunc" members of a classifier receive as input
a dataset and the id of a pattern rather than the pattern itself.


A DataSet now supports kernel/similarity data by:
>>> d = datafunc.DataSet(K=K, L = L)
where K is a Numeric 2-d matrix, and L is a list of labels (optional).
Construction of a dataset from a Numeric array X is now:
>>> d = datafunc.DataSet(X=X, L = L)

This help support usage of precomputed kernels:
If you install William Stafford Noble's Gist package, you can train and
test SVMs on any kernel matrix that was computed in advance.

Changes in the "Result" class; also added better support for saving objects
so that they are recoverable even when changes are made in the code.

Notes:

The Platt classifier (in composite.py) - is basically a translation of the spider
code; it suffers from some numerical instability that sometimes makes
it break down.


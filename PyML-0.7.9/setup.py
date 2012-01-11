#!/usr/bin/env python

"""Setup script for PyML."""

import os, sys
from distutils.core import setup
from distutils.extension import Extension
from distutils.command.build_ext import build_ext

if not hasattr(sys, 'version_info') or sys.version_info < (2,5,0,'alpha',0):
    raise SystemExit, "Python 2.5 or later is required."

try :
    import numpy
except :
    raise SystemExit, "The numpy package is required"
if numpy.version.version < '1.0' :
    raise SystemExit, 'numpy 1.0 or later is required'

name = "PyML"
version = "0.7.9"

PyML_packages = ['PyML', 'PyML/base',
                 'PyML/containers', 'PyML/containers/ext', 
                 'PyML/classifiers', 'PyML/classifiers/ext',
                 'PyML/classifiers/liblinear',
                 'PyML/clusterers', 'PyML/clusterers/ext',
                 'PyML/evaluators', 'PyML/preproc', 'PyML/feature_selection',
                 'PyML/datagen', 'PyML/demo',
                 'PyML/utils', 'PyML/utils/ext']

include_dirs = []

extensions = [
    Extension(name = 'PyML/containers/ext/_csparsedataset',
              sources = ['PyML/containers/ext/SparseDataSet_wrap.cpp',
                         'PyML/containers/ext/SparseDataSet.cpp',
                         'PyML/containers/ext/DataSet.cpp',
                         'PyML/containers/ext/SparseFeatureVector.cpp',
                         'PyML/containers/ext/Kernel.cpp'],
              include_dirs = include_dirs),
    Extension(name = 'PyML/containers/ext/_cvectordataset',
              sources = ['PyML/containers/ext/VectorDataSet_wrap.cpp',
                         'PyML/containers/ext/VectorDataSet.cpp',
                         'PyML/containers/ext/DataSet.cpp',
                         'PyML/containers/ext/FeatureVector.cpp',
                         'PyML/containers/ext/Kernel.cpp'],
              include_dirs = include_dirs),
    Extension(name = 'PyML/containers/ext/_csequencedata',
              sources = ['PyML/containers/ext/SequenceData_wrap.cpp',
                         'PyML/containers/ext/SequenceData.cpp',
                         'PyML/containers/ext/DataSet.cpp',
                         'PyML/containers/ext/Kernel.cpp'],
              include_dirs = include_dirs),

    Extension(name = 'PyML/containers/ext/_cpairdataset',
              sources = ['PyML/containers/ext/PairDataSet_wrap.cpp',
                         'PyML/containers/ext/PairDataSet.cpp',
                         'PyML/containers/ext/DataSet.cpp',
                         'PyML/containers/ext/Kernel.cpp'],
              include_dirs = include_dirs),
    
    Extension(name = 'PyML/containers/ext/_csetdata',
              sources = ['PyML/containers/ext/SetData_wrap.cpp',
                         'PyML/containers/ext/SetData.cpp',
                         'PyML/containers/ext/DataSet.cpp',
                         'PyML/containers/ext/Kernel.cpp'],
              include_dirs = include_dirs),
    
    Extension(name = 'PyML/containers/ext/_caggregate',
              sources = ['PyML/containers/ext/Aggregate_wrap.cpp',
                         'PyML/containers/ext/Aggregate.cpp',
                         'PyML/containers/ext/DataSet.cpp',
                         'PyML/containers/ext/Kernel.cpp'],
              include_dirs = include_dirs),              
    Extension(name = 'PyML/containers/ext/_ckerneldata',
              sources = ['PyML/containers/ext/KernelData_wrap.cpp',
                         'PyML/containers/ext/KernelData.cpp',
                         'PyML/containers/ext/DataSet.cpp',
                         'PyML/containers/ext/Kernel.cpp'],
              include_dirs = include_dirs),
    Extension(name = 'PyML/containers/ext/_ckernel',
              sources = ['PyML/containers/ext/Kernel_wrap.cpp',
                         'PyML/containers/ext/Kernel.cpp'],
              include_dirs = include_dirs),
    Extension(name = 'PyML/classifiers/ext/_libsvm',
              sources = ['PyML/classifiers/ext/libsvm_wrap.cpp',
                         'PyML/classifiers/ext/libsvm.cpp'],
              include_dirs = include_dirs),
    Extension(name = 'PyML/classifiers/ext/_mylibsvm',
              sources = ['PyML/classifiers/ext/mylibsvm_wrap.cpp',
                         'PyML/classifiers/ext/mylibsvm.cpp'],
              include_dirs = include_dirs),
    Extension(name = 'PyML/classifiers/ext/_csmo',
              sources = ['PyML/classifiers/ext/SMO_wrap.cpp',
                         'PyML/classifiers/ext/SMO.cpp',
                         'PyML/classifiers/ext/KernelCache.cpp',
                         'PyML/containers/ext/DataSet.cpp',
                         'PyML/containers/ext/Kernel.cpp'],
              include_dirs = include_dirs),
    Extension(name = 'PyML/classifiers/ext/_cgist',
              sources = ['PyML/classifiers/ext/Gist_wrap.cpp',
                         'PyML/classifiers/ext/Gist.cpp',
                         'PyML/classifiers/ext/KernelCache.cpp',
                         'PyML/containers/ext/DataSet.cpp',
                         'PyML/containers/ext/Kernel.cpp'],
              include_dirs = include_dirs),
    Extension(name = 'PyML/classifiers/liblinear/_mylinear',
              sources = ['PyML/classifiers/liblinear/mylinear_wrap.cpp',
                         'PyML/classifiers/liblinear/mylinear.cpp',
                         'PyML/containers/ext/SparseDataSet.cpp',
                         'PyML/containers/ext/DataSet.cpp',                         
                         'PyML/containers/ext/SparseFeatureVector.cpp',],
              include_dirs = include_dirs),
    Extension(name = 'PyML/classifiers/ext/_csvmodel',
              sources = ['PyML/classifiers/ext/SVModel_wrap.cpp',
                         'PyML/classifiers/ext/SVModel.cpp',
                         'PyML/containers/ext/Kernel.cpp',
                         'PyML/containers/ext/SparseDataSet.cpp',
                         'PyML/containers/ext/VectorDataSet.cpp',
                         'PyML/containers/ext/DataSet.cpp',
                         'PyML/containers/ext/SparseFeatureVector.cpp',
                         'PyML/containers/ext/FeatureVector.cpp'],
              include_dirs = include_dirs),
    Extension(name = 'PyML/classifiers/ext/_knn',
              sources = ['PyML/classifiers/ext/KNN_wrap.cpp',
                         'PyML/classifiers/ext/KNN.cpp',
                         'PyML/containers/ext/DataSet.cpp',
                         'PyML/containers/ext/Kernel.cpp',
                         #'PyML/ext/FeatureVector.cpp',
                         ],
              include_dirs = include_dirs),
    Extension(name = 'PyML/clusterers/ext/_ckmeans',
              sources = ['PyML/clusterers/ext/kmeans_wrap.cpp',
                         'PyML/clusterers/ext/kmeans.cpp',
                         'PyML/containers/ext/DataSet.cpp',
                         'PyML/containers/ext/Kernel.cpp',
                         #'PyML/ext/FeatureVector.cpp',
                         ],
              include_dirs = include_dirs),
    Extension(name = 'PyML/utils/ext/_carrayWrap',
              sources = ['PyML/utils/ext/arrayWrap_wrap.cpp'],
              include_dirs = include_dirs)
    ]

class build_ext_pyml(build_ext) :

    def run(self) :

        build_ext.run(self)

setup (name = name,
       version = version,
       description = "PyML - a Python machine learning package",
       #long_description = long_description
       author = "Asa Ben-Hur",
       author_email = "myfirstname@cs.colostate.edu",
       url = 'http://pyml.sourceforge.net',
       license = "GPL",
       cmdclass={"build_ext" : build_ext_pyml},
       #py_modules = PyML_modules,
       packages = PyML_packages,
       ext_modules = extensions,
       #extra_path = 'PyML'
       )

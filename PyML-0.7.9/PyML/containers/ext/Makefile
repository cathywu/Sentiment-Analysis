
SWIG=/usr/local/bin/swig
#SWIG=swig
#PYTHON_INCLUDEDIR=/usr/include/python2.4/
PYTHON_INCLUDEDIR=/Library/Frameworks/Python.framework/Versions/2.4/include/python2.4
PYTHON_INCLUDEDIR=/Library/Frameworks/Python.framework/Versions/2.5/include/python2.5/
PYTHON_INCLUDEDIR=/Library/Frameworks/Python.framework/Versions/2.6/include/python2.6/

#PYTHON_INCLUDEDIR=/s/chopin/c/proj/protfun/python/x86_64/include/python2.5

HEADERFILES=SparseDataSet.i SparseFeatureVector.i PairDataSet.i Aggregate.i KernelData.i Kernel.i VectorDataSet.i SequenceData.i StringKernel.i SetData.i

WRAPPERFILES=$(HEADERFILES:.i=_wrap.cpp)

wrap: $(WRAPPERFILES)
all: $(WRAPPERFILES)

%_wrap.cpp : %.i %.h
	$(SWIG) -python -shadow -c++ -o $@ $<

clean:
	rm -f *.o *.so *wrap*

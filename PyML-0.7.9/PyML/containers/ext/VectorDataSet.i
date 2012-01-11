%module cvectordataset
%{
	#include "DataSet.h"
	#include "VectorDataSet.h"
%}

%include "std_vector.i"

namespace std
{
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
   %template(StringVector) vector<string>;
   %template(LongVector) vector<long>;
}

%include "DataSet.h"
%include "VectorDataSet.h"


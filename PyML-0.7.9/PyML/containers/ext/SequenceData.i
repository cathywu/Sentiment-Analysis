%module csequencedata
%{
	#include "DataSet.h"
	#include "SequenceData.h"
%}

%include "std_vector.i"
%include "std_string.i"

namespace std
{
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
   %template(StringVector) vector<string>;
   %template(LongVector) vector<long>;
}

%include "DataSet.h"
%include "SequenceData.h"


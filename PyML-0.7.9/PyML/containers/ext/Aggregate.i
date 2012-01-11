%module caggregate
%{
#include "DataSet.h"
#include "Aggregate.h"
%}

%include "std_vector.i"

namespace std 
{
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
};

%include "DataSet.h"
%include "Aggregate.h"


%module ckmeans
%{
#include "kmeans.h"
%}

%include "std_vector.i"

namespace std 
{
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
}

%include "kmeans.h"


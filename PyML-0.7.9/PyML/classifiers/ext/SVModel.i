%module csvmodel
%{
#include "SVModel.h"
%}

%include "std_vector.i"

namespace std 
{
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
};

%include "SVModel.h"


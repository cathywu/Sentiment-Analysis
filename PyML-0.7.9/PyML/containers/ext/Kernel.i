%module ckernel
%{
#include "Kernel.h"
%}

%include "std_vector.i"
%include "std_string.i"
namespace std
{
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
   %template(FloatVector) vector<float>;
}

%include "Kernel.h"


%module csetdata
%{
#include "SetData.h"
#include "SetData.h"
%}

%include "std_vector.i"

namespace std 
{
    %template(IntVector) std::vector<int>;
    %template(DoubleVector) std::vector<double>;
};

%include "DataSet.h"
%include "SetData.h"

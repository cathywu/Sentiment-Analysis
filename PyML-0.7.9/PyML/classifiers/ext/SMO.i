%module csmo
%{
#include "SMO.h"
%}

%include "std_vector.i"

namespace std 
{
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
};


std::vector<double> runSMO(DataSet *data, 
			   const std::vector<double> &C,
			   //std::vector<double> &alpha,
			   int cacheSize);

%module cgist
%{
#include "Gist.h"
%}

%include "std_vector.i"

namespace std 
{
   %template(IntVector) vector<int>;
   %template(DoubleVector) vector<double>;
};


void runGist(DataSet *data, 
	     const std::vector<double> &C, 
	     std::vector<double> &alpha,
	     int cacheSize,
	     int iterations);

void runGradientDescent(DataSet *data, 
	     const std::vector<double> &C, 
	     std::vector<double> &alpha,
	     int cacheSize,
	     int iterations);

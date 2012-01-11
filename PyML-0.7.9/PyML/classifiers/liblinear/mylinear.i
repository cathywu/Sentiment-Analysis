%module mylinear
%{
#include "mylinear.h"
%}

%include "std_vector.i"

namespace std
{
   %template(IntVector) std::vector<int>;	
   %template(DoubleVector) std::vector<double>;
}

std::vector<double> solve_l2r_l1l2_svc(SparseDataSet *data, double eps, 
				       double Cp, double Cn, int solver_type);


%module carrayWrap
%{
%}

%include "std_vector.i"
%include "std_string.i"

namespace std
{
   %template(IntVector) std::vector<int>;	
   %template(DoubleVector) std::vector<double>;
   %template(FloatVector) std::vector<float>;
   %template(StringVector) std::vector<string>;
   %template(LongVector) std::vector<long>;
}

%include "carrays.i"
%array_class(int, intArray);
%array_class(double, doubleArray);
%pointer_class(DataSet, datasetPtr);


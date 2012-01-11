
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <stdlib.h>

//#include "SparseDataSet.h"
#include "Kernel.h"
//#include "SMO.h"

using namespace std;

void split(const string& str,
	   vector<string>& tokens,
	   const string& delimiter);

int main(int argc, char** argv) {
  
  Linear k;
  Kernel *k2;
  k2 = k.duplicate();

  delete k2;
  Gaussian k3;
  Gaussian *k4;
  k4 = k3.duplicate();
  delete k4;

}

void split(const string& str,
	   vector<string>& tokens,
	   const string& delimiter)
{
  // Skip delimiters at beginning.
  string::size_type lastPos = str.find_first_not_of(delimiter, 0);
  // Find first delimiter
  string::size_type pos     = str.find_first_of(delimiter, lastPos);
  
  while (string::npos != pos || string::npos != lastPos)
    {
      // Found a token, add it to the vector.
      tokens.push_back(str.substr(lastPos, pos - lastPos));
      // Skip delimiters.  Note the "not_of"
      lastPos = str.find_first_not_of(delimiter, pos);
      // Find next "non-delimiter"
      pos = str.find_first_of(delimiter, lastPos);
    }
}

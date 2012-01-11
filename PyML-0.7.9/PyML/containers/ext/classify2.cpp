
#include <iostream>
#include <string>
#include <vector>
#include <fstream>
#include <stdlib.h>

#include "SparseDataSet.h"
#include "Kernel.h"
#include "SMO.h"

using namespace std;

void split(const string& str,
	   vector<string>& tokens,
	   const string& delimiter);

int main(int argc, char** argv) {

  char *fileName;
  fileName = argv[1];

  ifstream infile(fileName);
  vector<string> tokens;
  string line;
  SparseDataSet data(270);

  int pattern = 0;
  while(getline(infile, line)) {
    cout << "pattern " << pattern << endl;
    tokens.clear();
    split(line, tokens, ",");
    double y = atof(tokens[0].c_str());
    if (y == -1) {
      data.setY(pattern, 0);
    }
    else {
      data.setY(pattern, 1);
    }

    vector<int> featureID;
    vector<double> featureValue;
    for (int i = 1; i < tokens.size(); ++i) {
      featureID.push_back(i);
      featureValue.push_back(atof(tokens[i].c_str()));
    }
    data.addPattern(pattern, featureID, featureValue);
    ++pattern;
  }
  data.featureIDcompute();
  data.show();
  data.kernel = new Cosine();
  
  cout << data.kernel->eval(&data, 0, 0) << endl;;

  std::vector<double> C(data.size(), 10.0);

  SMO s(data.castToBase(), C, 100);
  s.optimize();
  
  Linear k;
  Kernel *k2;
  k2 = k.duplicate();

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

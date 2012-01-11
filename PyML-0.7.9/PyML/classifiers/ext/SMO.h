# ifndef SMO_H
# define SMO_H

# include "../../containers/ext/DataSet.h"
# include "../../containers/ext/Kernel.h"
# include "KernelCache.h"

# include <set>
# include <ext/hash_set>
# include <vector>
# include <algorithm>
# include <functional>
#include <string>
#include <iostream>
#include <fstream>
# include <cmath>

#define INF HUGE_VAL

using namespace std;

using __gnu_cxx::hash_set;

typedef vector<int> IndexSet;
typedef vector<int>::iterator IndexSetItr;

class SMO {
 public:
  DataSet *data;
  std::vector<double> Y;
  std::vector<double> alpha;
  double b;
  std::vector<double> C;
  double eps;
  double tolerance;

  KernelCache cache;

  std::vector<double> Kdiag;
  bool isLinear;

  std::vector<double> G;
  std::vector<double> Gbar;

  IndexSet activeSet;
  bool shrinking;
  bool unshrinked;

  int size() { return data->size(); }
  bool isUpperBound(int i) { return alpha[i] >= C[i]; }
  bool isLowerBound(int i) { return alpha[i] <= 0; }
  bool isFree(int i) { return (alpha[i] > 0 && alpha[i] < C[i]);}

  bool selectWorkingSet(int &iOut, int &jOut);
  void update(int i, int j);
  void reconstructGradient();
  void shrink();
  void optimize();

  SMO(DataSet *_data, const std::vector<double> &C_, const int cacheSize);
  ~SMO();

  double compute_b();

  void show();

};

std::vector<double> runSMO(DataSet *data, 
			   const std::vector<double> &C,
			   int cacheSize);

# endif

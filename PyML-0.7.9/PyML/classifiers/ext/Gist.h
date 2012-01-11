# ifndef GIST_H
# define GIST_H

# include "../../containers/ext/SparseDataSet.h"
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


class Gist {
 public:
  DataSet *data;
  std::vector<double> Y;
  std::vector<double> alpha;
  double b;
  double objectiveValue;
  std::vector<double> C;
  int maxiter;
  double eps;
  double tolerance;
  double convergenceThreshold;

  KernelCache cache;

  //std::vector<double> Kdiag;
  bool isLinear;

  int size() { return data->size(); }
  bool isUpperBound(int i) { return alpha[i] >= C[i]; }
  bool isLowerBound(int i) { return alpha[i] <= 0; }
  bool isFree(int i) { return (alpha[i] > 0 && alpha[i] < C[i]);}

  double decisionFunc(int pattern, vector<float> &kernelRow);
  double objectiveFunction();

  bool optimize();
  bool converged();
  double updateAlpha(int pattern);


  Gist(DataSet *_data, 
       const std::vector<double> &C_, 
       const int cacheSize, 
       const int maxiter);

  ~Gist();

  void show();

};

void runGist(DataSet *data, 
	     const std::vector<double> &C,
	     std::vector<double> &alpha,
	     int cacheSize,
	     int iterations);

class GradientDescent {
 public:
  DataSet *data;
  std::vector<double> Y;
  std::vector<double> alpha;
  double b;
  double objectiveValue;
  std::vector<double> C;
  int maxiter;
  double eps;
  double tolerance;
  double convergenceThreshold;
  double learningRate;

  KernelCache cache;

  bool isLinear;

  int size() { return data->size(); }
  bool isUpperBound(int i) { return alpha[i] >= C[i]; }
  bool isLowerBound(int i) { return alpha[i] <= 0; }
  bool isFree(int i) { return (alpha[i] > 0 && alpha[i] < C[i]);}

  double decisionFunc(int pattern, vector<float> &kernelRow);
  double objectiveFunction();

  bool optimize();
  bool converged();
  double updateAlpha(int pattern);


  GradientDescent(DataSet *_data, 
       const std::vector<double> &C_, 
       const int cacheSize, 
       const int maxiter);

  ~GradientDescent();

  void show();

};

void runGradientDescent(DataSet *data, 
			const std::vector<double> &C,
			std::vector<double> &alpha,
			int cacheSize,
			int iterations);




# endif

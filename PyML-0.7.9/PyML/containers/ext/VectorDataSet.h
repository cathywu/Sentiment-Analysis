#ifndef VECTORDATASET_H
#define VECTORDATASET_H

# include <stdio.h>
# include <ctype.h>
# include <float.h>
# include <math.h>
//# include <cmath>

# include <iostream>
# include <algorithm>
# include <vector>
# include <string>
//# include <ext/hash_map>

//#include <stdlib.h>
//#include <stdarg.h>
//#include <string.h>

# include "FeatureVector.h"
# include "Kernel.h"
# include "../../classifiers/ext/libsvm.h"
# include "DataSet.h"

# define myint long
//using __gnu_cxx::hash_map;

using namespace std;

class VectorDataSet : public DataSet {
 private:
  //vector <long> featureID;
 public:
  int n;                 // number of examples
  int numFeatures;       // dimensionality of the data
  struct svm_problem prob;

  //the data vectors:
  vector<FeatureVector> X;    //the primary container, referenced by []

  vector <string> featureName;
  //hash_map <long,int> featureIDmap;
  
  VectorDataSet() {}
  VectorDataSet(const int n);
  VectorDataSet(const VectorDataSet &other, const std::vector<int> &patterns);
  ~VectorDataSet();
  
  FeatureVector& operator [] (int i) {return X[i];}

  int size() { return n; }
  DataSet* castToBase() { return dynamic_cast<DataSet *>(this); }
	
  vector<double> getPattern(int i);
  vector<double> getFeature(int j);
  
  void show();
  VectorDataSet* duplicate(const std::vector<int> &patterns)
  { return new VectorDataSet(*this, patterns); }

  void addPattern(const std::vector<double>& featureValue);
  void addFeature(const std::vector<double>& values);
  void addFeatures(VectorDataSet& other);
  void featureIDcompute();

  void setFeatureName(int i, string& fname);
  void eliminateFeatures(const std::vector<int>& featuresToEliminate);

  void normalize(int p);
  void scale(const std::vector<double> &w);
  void translate(const std::vector<double> &a);

  std::vector<double> mean(const std::vector<int>& patterns);
  std::vector<double> standardDeviation(const std::vector<int>& patterns);
  void weightedSum(FeatureVector& w, const std::vector<int>& patterns,
				   const std::vector<double>& alpha);

  int featureCount(int feature, const std::vector<int>& patterns);
  std::vector<int> featureCounts(const std::vector<int>& patterns);
  std::vector<int> nonzero(int feature, const std::vector<int>&patterns);
  //int featureSetCount(const std::vector<int>& features, const std::vector<int>& patterns);
	
  double dotProduct(int i, int j)
  { 
	double sum = 0;
	for (int k = 0; k < numFeatures; ++k) {
	  sum += X[i][k] * X[j][k];
	}
	return sum;
  }
	
  double dotProduct(int i, int j, DataSet* other)
  { 
	double sum = 0;
	VectorDataSet* other_ptr = dynamic_cast<VectorDataSet*>(other);
	for (int k = 0; k < numFeatures; ++k) {
	  sum += X[i][k] * other_ptr->X[j][k];
	}
	return sum;
  }

  void libsvm_construct(svm_problem &prob);
};

# endif






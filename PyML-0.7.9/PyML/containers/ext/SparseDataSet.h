#ifndef SPARSEDATASET_H
#define SPARSEDATASET_H

# include <stdio.h>
# include <ctype.h>
# include <float.h>
# include <math.h>
# include <cmath>

# include <iostream>
# include <algorithm>
# include <vector>
# include <string>
# include <ext/hash_map>

//#include <stdlib.h>
//#include <stdarg.h>

# include "SparseFeatureVector.h"
# include "Kernel.h"
# include "DataSet.h"
# include "../../classifiers/ext/libsvm.h"

# define myint long

using __gnu_cxx::hash_map;
//using std::hash_map

using namespace std;
//using namespace __gnu_cxx;

using std::vector;

class SparseDataSet : public DataSet {
 private:
  vector <long> featureID;

 public:
  int n;// number of examples
  int numFeatures;
  //int d;// dimensionality of the data
  struct svm_problem prob;

  hash_map <long,int> featureIDmap;
  std::vector<SparseFeatureVector> X;    //the data container, referenced by []

  vector <string> featureName;
  
  SparseDataSet() {}
  SparseDataSet(const int n);
  SparseDataSet(const SparseDataSet &other, const std::vector<int> &patterns);

  SparseFeatureVector& operator [] (int i) {return X[i]; }

  int size() { return n; }
  DataSet* castToBase() { return dynamic_cast<DataSet *>(this); }
	
  vector<long> getFeatureID() { return featureID; }
  vector<double> getPattern(int i);
  vector<double> getPatternValues(int i);
  vector<long> getPatternKeys(int i);
  void getPatternSparse(int i, std::vector<double> &values, std::vector<long> &ids);
  vector<double> getFeature(int j);

  void show();

  SparseDataSet* duplicate(const std::vector<int> &patterns) { 
	return new SparseDataSet(*this, patterns); }

  ~SparseDataSet();

  void addPattern(const std::vector<long>& featureID_, const std::vector<double>& featureValue);
  void addFeature(long id, const std::vector<double>& values);
  void addFeatures(SparseDataSet& other);

  void featureIDcompute(void);

  void setFeatureName(int i, string& fname);
  long getFeatureID(int i);
  void eliminateFeatures(const std::vector<int>& featuresToEliminate);

  void normalize(int p);
  void scale(const std::vector<double> &w);
  void translate(const std::vector<double> &a);

  std::vector<double> mean(const std::vector<int>& patterns);
  std::vector<double> standardDeviation(const std::vector<int>& patterns);

  void weightedSum(SparseFeatureVector& w, const std::vector<int>& patterns,
				   const std::vector<double>& alpha);

  std::vector<int> commonFeatures(int pattern1, int pattern2);
  int featureCount(int feature, const std::vector<int>& patterns);
  std::vector<int> featureCounts(const std::vector<int>& patterns);
  std::vector<int> nonzero(int feature, const std::vector<int>&patterns);
  //  int featureSetCount(const std::vector<int>& features, const std::vector<int>& patterns);

  double dotProduct(int i, int j, DataSet* other)
  { 
	return X[i].dotProduct( dynamic_cast<SparseDataSet*>(other)->X[j]); 
  }
  double dotProduct(int i, int j)
  { 
	return X[i].dotProduct(X[j]);
  }

  void libsvm_construct(struct svm_problem &prob);

};


# endif

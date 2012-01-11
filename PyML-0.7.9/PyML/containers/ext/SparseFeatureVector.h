# ifndef SPARSEFEATUREVECTOR_H
# define SPARSEFEATUREVECTOR_H

# include <list>
# include <iostream>
# include <vector>
# include <math.h>
# include <cmath>

class Feature {
 public:

  long index;
  double value;
  
  Feature(const Feature& other);
  Feature(long featureID, double featureValue);
  bool compare(Feature &other) { return (index < other.index); }
};

using namespace std;

typedef list<Feature>::iterator featureIterator;

class SparseFeatureVector {
 public:
  list<Feature> features;
  
  SparseFeatureVector();
  SparseFeatureVector(vector<long> featureIDs, vector<double> featureValues);
  SparseFeatureVector(const SparseFeatureVector& other);
				
  void initialize(vector<long> featureIDs, vector<double> featureValues);
  void get(vector<double>& values, vector<long>& ids);
  vector<double> getValues(void);
  vector<long> getKeys(void);
  void add(long id, double value);
  void add(SparseFeatureVector& other);

  void printData();
  
  vector<int> commonFeatures(SparseFeatureVector& other);
  void scale(double w);
  double norm(int q);
  double dotProduct(SparseFeatureVector& other);
  int size();
	  
};
  
# endif

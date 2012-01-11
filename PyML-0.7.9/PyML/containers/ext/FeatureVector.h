//FeatureVector
# ifndef FEATUREVECTOR_H
# define FEATUREVECTOR_H

# include <iostream>
# include <vector>
# include <math.h>
# include <cmath>

using namespace std;

typedef vector<double>::iterator vectorIterator;

class FeatureVector {
 public:
  vector<double> features;
		
  FeatureVector();
  FeatureVector(vector<double> featureValues);
  FeatureVector(const FeatureVector& other);
  
  inline double& operator [] (int i) {return features[i];}

  void initialize(vector<double> featureValues);
  void get(vector<double>& values);
  void add(double value);
  void add(FeatureVector& other);

  void erase(vector<bool>& featureIndicator);
  void clear();
		
  void printData();
		
  void scale(double w);
  double norm(int q);
  double dotProduct(FeatureVector& other);
  int size();

};

# endif

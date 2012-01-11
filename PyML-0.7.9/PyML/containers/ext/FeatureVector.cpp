//FeatureVector
# include "FeatureVector.h"
# include <cmath>

FeatureVector::FeatureVector()
{ }

FeatureVector::FeatureVector(const FeatureVector& other) :
  features(other.features)
{ }

FeatureVector::FeatureVector(vector<double> featureValues) :
  features(featureValues)
{ }

void FeatureVector::initialize(std::vector<double> featureValue)
{
  for (long int i = 0; i < featureValue.size(); i++) {
	features.push_back(featureValue[i]);
  }
}

void FeatureVector::get(vector<double>& values)
{
  values.reserve(size());
  for (long int j = 0; j <features.size(); j++) {
    values.push_back(features[j]);
  }
}

void FeatureVector::add(double value)
{
  features.resize(size() + 1, value);
}

void FeatureVector::add(FeatureVector& other)
{
  features.insert(features.end(), other.features.begin(), other.features.end());
}

void FeatureVector::erase(vector<bool>& featureIndicator)
{
  vector<double> x;
  for (int i = 0; i < size(); ++i) {
	if (featureIndicator[i]) {
	  x.push_back(features[i]);
	}
  }
  features = x;
}

void FeatureVector::clear()
{
	features.clear();
}

void FeatureVector::printData() 
{
  for (int j = 0; j < features.size(); j++) {
    cout << j << ":" << features[j] << " ";
  }
  cout << "\n";
}

void FeatureVector::scale (double w)
{
  for(vectorIterator xiter = features.begin(); xiter != features.end(); ++xiter){
    (*xiter) *= w;
  }
}

double FeatureVector::norm (int p)
{

  double sum = 0;
  vectorIterator xiter;

  xiter = features.begin();
  for(vectorIterator xiter = features.begin();  xiter != features.end(); ++xiter) 
	{
    if (p == 2) {
      sum += (*xiter) * (*xiter);
    }
    else {
      sum += abs((*xiter));
    }
  }
  
  if (p == 2) {
  	return sqrt(sum);
  }
  else {
    return sum;
  }
}


double FeatureVector::dotProduct (FeatureVector& other)
{
  double sum = 0;
  for (int i = 0; i < size(); ++i) {
	sum += features[i] * other.features[i];
  }

  return sum;

}

int FeatureVector::size()
{
  return features.size();
}





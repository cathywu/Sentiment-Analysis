# include "SparseFeatureVector.h"
# include <cmath>

Feature::Feature(long featureID, double featureValue) :
  index(featureID),
  value(featureValue)
{ }

Feature::Feature(const Feature& other) :
  index(other.index),
  value(other.value)
{ }

SparseFeatureVector::SparseFeatureVector() { }

SparseFeatureVector::SparseFeatureVector(std::vector<long> featureID,
					 std::vector<double> featureValue)
{
  if (featureID.size() == 0) {
    for (unsigned int i = 0; i < featureValue.size(); i++) {
      features.push_back(Feature(i, featureValue[i]));
    }
  }
  else {
    for (unsigned int i = 0; i < featureValue.size(); i++) {
      features.push_back(Feature(featureID[i], featureValue[i]));
    }
  }
}

SparseFeatureVector::SparseFeatureVector(const SparseFeatureVector& other)
{
  features = other.features;
}

void SparseFeatureVector::initialize(std::vector<long> featureID,
				     std::vector<double> featureValue)
{
  for (unsigned int i = 0; i < featureID.size(); i++) {
    features.push_back(Feature(featureID[i], featureValue[i]));
  }
}

void SparseFeatureVector::add(long index, double value)
{
  if (value == 0) {
    return;
  }
  for (featureIterator fiter = features.begin(); fiter != features.end(); ++fiter){
    if (index < (*fiter).index ) {
      features.insert(fiter, Feature(index, value));
      return;
    }
  }
  features.insert(features.end(), Feature(index, value));
}

void SparseFeatureVector::add(SparseFeatureVector& other)
{
  featureIterator xiter, yiter;
  xiter = features.begin();
  yiter = other.features.begin();
  while(xiter != features.end() && yiter != other.features.end()) {
    if((*xiter).index < (*yiter).index){
      ++xiter;
    }
    else{
	  features.insert(xiter, *yiter);
	  ++yiter;
    }		
  }
  // if there is anything left in the other feature vector, add them at the end:
  if (yiter != other.features.end()) {
      while(yiter != other.features.end()) {
	  features.insert(xiter, *yiter);
	  ++yiter;
      }
  }
}
vector<double> SparseFeatureVector::getValues(void)
{
    vector<double> values(size());
    for (featureIterator j = features.begin(); j != features.end(); j++) {
	values.push_back(j->value);
    }
    return values;
}
vector<long> SparseFeatureVector::getKeys(void)
{
    vector<long> keys(size());
    for (featureIterator j = features.begin(); j != features.end(); j++) {
	keys.push_back(j->index);
    }
    return keys;

}

void SparseFeatureVector::get(vector<double> &values, vector<long> &ids)
{
  values.reserve(size());
  ids.reserve(size());
  for (featureIterator j = features.begin(); j != features.end(); j++) {
      values.push_back(j->value);
      ids.push_back(j->index);
  }
}

void SparseFeatureVector::printData(void) {

  for (featureIterator j = features.begin(); j != features.end(); j++) {
    cout << (*j).index << ":" << (*j).value << " ";
  }
  cout << "\n";
}

void SparseFeatureVector::scale (double w)
{
  for(featureIterator xiter= features.begin(); 
      xiter != features.end(); 
      xiter++){
    (*xiter).value *= w;
  }
}

double SparseFeatureVector::norm (int p)
{
  double sum = 0;
  featureIterator xiter;

  xiter = features.begin();
  for(featureIterator xiter = features.begin(); xiter != features.end(); ++xiter){
    if (p == 2) {
      sum += (*xiter).value * (*xiter).value;
    }
    else {
      sum += abs((*xiter).value);
    }
  }
  if (p == 2) {
    return sqrt(sum);
  }
  else {
    return sum;
  }
}

double SparseFeatureVector::dotProduct (SparseFeatureVector& other)
{
  double sum = 0;
  featureIterator xiter, yiter;

  xiter = features.begin();
  yiter = other.features.begin();
  while(xiter != features.end() && yiter != other.features.end()) {
    if((*xiter).index == (*yiter).index){
      sum += (*xiter).value * (*yiter).value;
      ++xiter;
      ++yiter;
    }
    else{
      if((*xiter).index > (*yiter).index)
	++yiter;
      else
	++xiter;
    }			
  }
  return sum;
}
std::vector<int> SparseFeatureVector::commonFeatures(SparseFeatureVector &other)
{	
  vector<int> common;
  featureIterator xiter, yiter;

  xiter = features.begin();
  yiter = other.features.begin();
  while(xiter != features.end() && yiter != other.features.end()){
    if((*xiter).index == (*yiter).index){
      common.push_back((*xiter).index);
      ++xiter;
      ++yiter;
    }
    else{
      if((*xiter).index > (*yiter).index)
	++yiter;
      else
	++xiter;
    }
  }
  return common;
}

int SparseFeatureVector::size()
{
  return features.size();
}





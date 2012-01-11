//testDataSet
#import <stdio.h>
#import <iostream>
#include "Feature.cpp"
#include "SparseDataSet.h"
#include "VectorDataSet.h"

using std::vector;
using std::cout;
using std::endl;
using std::cerr;

SparseDataSet* data;

void test_nonzero()
{
	vector<int> patterns;
	for(int i = 0; i < 3; i++)
		patterns.push_back(i);

	vector<int> answers = data->nonzero( 2, patterns);
	for(int i = 0; i < answers.size(); i++)
	  cout << answers[i];
}


void buildData(vector<long> featureIDS, vector<double> values)
{
	data = new SparseDataSet(10);
	data->addPattern(featureIDS, values);
	data->addPattern(featureIDS, values);
	data->addPattern(featureIDS, values);
	cout << endl;
	data->show();
}

void test_weighted()
{
  FeatureVector result();
  vector<int> patterns;
	for(int i = 0; i < 3; i++)
	{
		const int temp = i;
		patterns.push_back(temp);
	}
	vector<double> alpha;
	for(int i = 4; i < 7; i++)
		alpha.push_back((double)i);
  //data->weightedSum(result, patterns, alpha);

}

/*double test_Iterator()
{
	list<Feature>::iterator iter = (data->X[0]->begin());
//	Feature* ptr = (data->X[0]->begin());
	Feature* retrieved = &(*iter);
	for(int i = 0; i < 20; i++)
	{
		cout << retrieved->value << " ";
    retrieved+=2;
	}
	return (retrieved->value);
}*/

int main(int argc, char** argv)
{
	cout << "working 1"<< endl;
	int size = 25;
	vector<long> featureIDS;
	vector<double> values;
	for(int i = 0; i< size; i++)
	{
		long index = i;
		featureIDS.push_back(index);
		double value = (i+1)*(i+1);
		values.push_back(value);
	}
	cout << "working 2"<< endl;
	FeatureVector testSet(featureIDS, values);
		cout << "working 3"<< endl;
  testSet.add((long) size, (double) 42);
  	cout << "working 4"<< endl;
	testSet.printData();
	cout << "working 5"<< endl;
	buildData(featureIDS, values);
	
	test_nonzero();
		cout << "working 6"<< endl;
	test_weighted();
	
	SparseDataSet* sparse = new SparseDataSet(10);

	//cout << (test_Iterator()) ;

	char pause;
	cin >> pause;
	return 0;
}

/*
nonzero( 2, patterns)
SparseDataSet(10);
addPattern(featureIDS, values)
FeatureVector()
FeatureVector testSet(featureIDS, values);
add((long) size, (double) 42)
printData()
SparseDataSet(10)
*/

/*
Method Coverage:
a * in front of the method means that it has been accessed by the test program.
a - means that it gets accessed by another method that is called by testing.

******FeatureVector*****
		FeatureVector();
		FeatureVector(vector<long> featureIDs, vector<double> featureValues);
		FeatureVector(const BaseFeatureVector& other);
		
		inline double& operator [] (int i) {return features[i];}
		BaseIterator& begin();
		BaseIterator& end();

		void insert(vector<long> featureIDs, vector<double> featureValues);
		void get(vector<double>& values, vector<long>& ids);
		void add(long id, double value);
		void erase(BaseIterator& iter);
		
		void printData();
		
		vector<int> commonFeatures(BaseFeatureVector& other);
		void scale(double w);
		double norm(int q);
		double dotProduct(BaseFeatureVector& other);
		int size();

******VectorDataSet*******
	VectorDataSet() {}
*	VectorDataSet(const int n);
  VectorDataSet(const VectorDataSet &other, const std::vector<int> &patterns);
	~VectorDataSet();
  
	FeatureVector& operator [] (int i) {return X[i];}

  vector<double> getPattern(int i);
  void getPatternSparse(int i, std::vector<double> &values, std::vector<long> &ids);
  vector<double> getFeature(int j);
  
  void convert2libsvm(void);
  void libsvmDestroy(void);
*	void show();
  VectorDataSet* duplicate(const std::vector<int> &patterns)
    { return new VectorDataSet(*this, patterns); }
*	void addPattern(vector<long>& featureID_, vector<double>& featureValue);
  void addFeature(long id, std::vector<double>& values);

  void featureIDcompute(void);
 	void setFeatureName(int i, string& fname);
  long getFeatureID(int i);
	void eliminateFeatures(const std::vector<int>& featuresToEliminate);

  void normalize(int p);
  void scale(const std::vector<double> &w);
  void translate(const std::vector<double> &a);

  std::vector<double> mean(const std::vector<int>& patterns);
  std::vector<double> standardDeviation(const std::vector<int>& patterns);
  void weightedSum(FeatureVector& w, const std::vector<int>& patterns,
				   const std::vector<double>& alpha);

  std::vector<int> commonFeatures(int pattern1, int pattern2);
  int featureCount(int feature, const std::vector<int>& patterns);
  std::vector<int> featureCounts(const std::vector<int>& patterns);
  std::vector<int> nonzero(int feature, const std::vector<int>&patterns);
  int featureSetCount(const std::vector<int>& features, const std::vector<int>& patterns);

  double dotProduct(int i, int j, DataSet* other = 0)
*/

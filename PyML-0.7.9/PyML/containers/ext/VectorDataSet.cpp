# include "VectorDataSet.h"

VectorDataSet::VectorDataSet(const int numPatterns) :
  DataSet(numPatterns)
{
  numFeatures = 0;
  n = numPatterns;
}

VectorDataSet::VectorDataSet(const VectorDataSet &other, const std::vector<int> &patterns)
 : DataSet(other, patterns)
{
  numFeatures = (other.numFeatures);
  n = (patterns.size());
  X.reserve(patterns.size());
  for (unsigned int i = 0; i < patterns.size(); i++){
	int p = patterns[i];
    X.push_back(other.X[p]);
    Y[i] = other.Y[p];
  }
}

VectorDataSet::~VectorDataSet()
{
  //cout << "in VectorDataSet::~VectorDataSet" << endl;
}

void VectorDataSet::featureIDcompute() 
{
  numFeatures = X[0].size();
}

void VectorDataSet::show() 
{
  cout << "VectorDataSet:" << endl;
  for (int i=0; i < 2; i++) {
    cout << i << " class : " << Y[i];
    cout << " x ";
    X[i].printData();
  }
  cout << "Number of Features: " << numFeatures << endl;

}

void VectorDataSet::setFeatureName(int i, string& fname)//identical to SparseDataSet
{
  featureName.push_back(fname);
}

//J: this method deserves a second look, refactoring could help
void VectorDataSet::eliminateFeatures(const std::vector<int> &featuresToEliminate)
{

  //sort(featuresToEliminate.begin(), featuresToEliminate.end());//make sure it's in order

  vector<bool> featureIndicator(numFeatures, true);  // which features to keep
  for (unsigned int j = 0; j < featuresToEliminate.size(); ++j) {
    featureIndicator[featuresToEliminate[j]] = false;
  }

  for (int i = 0; i < n; ++i) {
	X[i].erase(featureIndicator);
  }
  numFeatures -= featuresToEliminate.size();

}

void VectorDataSet::addPattern(const std::vector<double>& featureValue)
{
  X.push_back(FeatureVector(featureValue));
}

void VectorDataSet::addFeature(const std::vector<double>& values)
{
  if( values.size() != X.size()){
	cerr << "ERROR: The number of values provided does not match the number of vectors" << endl;
	return;
  }
  
  for (unsigned int i = 0; i < values.size(); ++i) {
	cout << "i = " << i << endl;
    X[i].add(values[i]);
  }
  numFeatures += 1;
}

void VectorDataSet::addFeatures(VectorDataSet& other)
{
  for (int i = 0; i < size(); ++i) 
  {
    X[i].add(other.X[i]);
  }
  numFeatures += other.numFeatures;
}

vector<double> VectorDataSet::getPattern(int i)
{
  vector<double> values;
  X[i].get(values);
  return values;

}

vector<double> VectorDataSet::getFeature(int j)
{
  vector<double> f(size(), 0);
  for (int i = 0; i < size(); ++i){
	f[i] = X[i][j];
  }
  return f;
}

void VectorDataSet::scale(const std::vector<double>& w)
{
  if (w.size() != numFeatures)
	{
    cout << "weight vector size " << w.size() << "dimension: " << numFeatures << endl;
    printf("wrong size of scaling vector\n");
    exit(128);
    return;
  }
  for (long i = 0; i < n; i++) {
    for (long j = 0; j < X[i].size(); ++j){
      X[i][j] *= w[j];
    }
  }
}

void VectorDataSet::translate(const std::vector<double> &a)
{
  for (int i = 0; i<n; i++){
    for (long j = 0; j < X[i].size(); ++j){
      X[i][j] -= a[j];
    }
  }
}

vector<double> VectorDataSet::mean(const std::vector<int>& patterns)
{
  int pIndex;

  std::vector<double> means(numFeatures, 0);//vector of means of each feature
  for (unsigned int i = 0; i < patterns.size(); i++)//for each pattern listed
  {
    pIndex = patterns[i];

    for (long j = 0; j < X[pIndex].size(); ++j)//run through the pattern
    {
      means[j] += X[pIndex][j];
    }
  }
  for (int j = 0; j < numFeatures; ++j) {//divide the totals
    means[j] /= float(patterns.size());
  }

  return means;
}

void VectorDataSet::weightedSum(FeatureVector& result, const std::vector<int> &patterns,
								const std::vector<double> &alpha)
{
  int pIndex;
  // ASA FeatureVector is an alias for vector<double>
  vector<double> weight_temp (numFeatures,0);

  for (unsigned int i = 0; i < patterns.size(); i++){
    pIndex = patterns[i];
    for (int j = 0; j != X[pIndex].size(); ++j){
      weight_temp[j] += X[pIndex][j] * alpha[i];
    }
  }
  result.clear();
  result.initialize(weight_temp);
  
}

std::vector<double>
VectorDataSet::standardDeviation(const std::vector<int>& patterns)
{
  int pIndex;
  std::vector<double> m = mean(patterns);
  std::vector<double> stdev(numFeatures, 0);
 
  // avg[n] = avg[n-1] - (avg[n-1] - xn) / n

  for (unsigned int i = 0; i < patterns.size(); i++){
    pIndex = patterns[i];
    for (int j = 0; j < X[pIndex].size(); j++) {
      double value = (X[pIndex][j] - m[j]) * (X[pIndex][j] - m[j]);
      stdev[j] = stdev[j] - (stdev[j] - value) / (i + 1);
    }
  }
  for (unsigned int j = 0; j < stdev.size(); j++){
    stdev[j] = sqrt(stdev[j]);
  }
  return stdev;

}

std::vector<int> VectorDataSet::featureCounts(const std::vector<int>& patterns)
{
  int pIndex;
  std::vector<int> counts(numFeatures);

  for (unsigned int i = 0; i < patterns.size(); i++){
    pIndex = patterns[i];
    for (long j = 0; j < X[pIndex].size(); ++j){
      if (X[pIndex][j] != 0.0)
	counts[j]++;
    }
  }
  return counts;
}

int VectorDataSet::featureCount(int featureToCount, const std::vector<int>& patterns)
{
  int pIndex;
  int count = 0;
  for (unsigned int i = 0; i < patterns.size(); i++){
    pIndex = patterns[i];
    if (X[pIndex][featureToCount] != 0.0)
      {
	++count;
      }
  }
  return count;
}


std::vector<int> VectorDataSet::nonzero(int feature, const std::vector<int>& patterns)
{
  int pIndex;
  std::vector<int> nonZeros;

  for (unsigned int i = 0; i < patterns.size(); i++)
	{
    pIndex = patterns[i];
		if (X[pIndex][feature] != 0.0)
		{
			nonZeros.push_back(pIndex);
    }
  }
  return nonZeros;
}

void VectorDataSet::normalize(int pIndex)
{
  for (int i = 0; i < n; i++)
	{
    double norm = X[i].norm(pIndex);
    if (norm > 0)
		{
		  X[i].scale(1.0/norm);
		}
	}
}

void VectorDataSet::libsvm_construct(svm_problem &prob) 
{
  prob.l = n;
  prob.x = new struct svm_node*[n];
  prob.y = new double[n];

  for (int i = 0; i < n; i++) 
	{
    prob.y[i] = double(Y[i]);
    unsigned int length = X[i].size();
    prob.x[i] = new svm_node[length + 1];
    prob.x[i][length].index = -1;
    prob.x[i][length].value = 0;
    
    for (int j = 0; j < X[i].size(); ++j) 
		{
      prob.x[i][j].index = j + 1;
      prob.x[i][j].value = X[i][j];

    }
  }

}






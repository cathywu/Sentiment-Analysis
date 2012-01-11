# include "SparseDataSet.h"

SparseDataSet::SparseDataSet(const int numPatterns) :  
  DataSet(numPatterns)
{ 
  n = (numPatterns);
  //X.reserve(numPatterns);
}

SparseDataSet::SparseDataSet(const SparseDataSet &other, const std::vector<int> &patterns) : 
  DataSet(other, patterns)
{
  featureID = (other.featureID);
  featureIDmap = (other.featureIDmap);
  numFeatures = other.numFeatures;

  n = (patterns.size());
  X.reserve(patterns.size());
  for (unsigned int i = 0; i < patterns.size(); i++) {
    int p = patterns[i];
    X.push_back(SparseFeatureVector(other.X[p]));
    Y[i] = other.Y[p];
  }
  //featureIDcompute();
  //attachKernel(other);
}

SparseDataSet::~SparseDataSet()
{
  //cout << "in SparseDataSet::~SparseDataSet" << endl;
}


void SparseDataSet::show() {

  cout << "SparseDataSet:" << endl;
  for (int i=0; i < 1; i++){
    cout << i << " class : " << Y[i];
    cout << " x ";
    X[i].printData();
  }
  cout << "Number of Features: " << numFeatures << endl;
  cout << "FeatureIDs : ";
  for (int i = 0; i < numFeatures; i++) {
    cout << " " << featureID[i];
  }
  cout << endl;
  if (norms.size() > 0) {
	cout << "Norms:  " << endl;
	for (int i = 0; i < 3; i++) {
	  cout << " " << norms[i];
	}
	cout << endl;
  }
}

void SparseDataSet::eliminateFeatures(const std::vector<int> &featuresToEliminate)
{
  bool *featureTable;  // the list of features that will remain
  featureTable = new bool[numFeatures];

  for (int j = 0; j < numFeatures; j++) {
    featureTable[j] = true;
  }
  for (unsigned int j = 0; j < featuresToEliminate.size(); j++) {
    featureTable[featuresToEliminate[j]] = false;
  }

  for (int i = 0; i < n; i++) {
    featureIterator j = X[i].features.begin();
    featureIterator temp;
    while (j != X[i].features.end()){
      temp = j;
      j++;
      if (!featureTable[featureIDmap[(*(temp)).index]]) {
	  X[i].features.erase(temp);
      }
    }
  }

  numFeatures -= featuresToEliminate.size();

  // erase from the featureIDmap all the ids that were eliminated:
  for (unsigned int i = 0; i < featuresToEliminate.size(); i++) {
    featureIDmap.erase(featureID[featuresToEliminate[i]]);
  }

  featureID.clear();
  //featureID.reserve(d);
    
  //copy them to the feature vector:
  for (hash_map<long,int>::iterator j = featureIDmap.begin();  
       j != featureIDmap.end();  j++){
      featureID.push_back((*j).first);
  }
  
  //create a new mapping by sorting the featureIDs and reassigning the 
  //featureIDmap
  sort(featureID.begin(), featureID.end());

  for (int i = 0; i < numFeatures; i++) {
      featureIDmap[featureID[i]] = i;
  }
}

void SparseDataSet::addPattern(const std::vector<long>& featureID_,
			       const std::vector<double>& featureValue)
{
    X.push_back(SparseFeatureVector(featureID_, featureValue));
}

void SparseDataSet::addFeature(long id, const std::vector<double>& values)
{
  for (int i = 0; i < size(); ++i) {
    X[i].add(id, values[i]);
  }
  featureIDcompute();
}

void SparseDataSet::addFeatures(SparseDataSet& other)
{
  for (int i = 0; i < size(); ++i){
	X[i].add(other.X[i]);
  }
  featureIDcompute();
}

void SparseDataSet::featureIDcompute(void) 
{
  long index;
  //accumulate the feature IDs in the data
  for (int i = 0; i < n; i++) {
    for (featureIterator j = X[i].features.begin(); j != X[i].features.end(); ++j) {
      index = (*j).index;
      hash_map<long, int>::iterator location = featureIDmap.find(index);
      if (location == featureIDmap.end()) {
	  featureIDmap[index] = 1;
	  featureID.push_back(index);
      }
    }
  }

  numFeatures = featureIDmap.size();
  sort(featureID.begin(), featureID.end());

  /*for (int i = 0; i < featureID.size(); i++) {
    cout << "i " << i << " featureID: " << featureID[i] << "\n";
  }

  cout << "featureIDmap size: " << featureIDmap.size() << "\n";

  cout << "size : " << featureID.size() << "\n"; */

  for (int i = 0; i < numFeatures; i++) {
    featureIDmap[featureID[i]] = i;
  }
  
  featureName.clear();
  featureName.reserve(numFeatures);

}

void SparseDataSet::setFeatureName(int i, string& fname)//identical to VectorDataSet
{
  featureName.push_back(fname);
}

long SparseDataSet::getFeatureID(int i)//identical to VectorDataSet
{
  return featureID[i];
}

vector<double> SparseDataSet::getPattern(int i)
{
  vector<double> x(numFeatures, 0);
  vector<double> values;
  vector<long> ids;
  X[i].get(values, ids);
  for (unsigned int j = 0; j < ids.size(); j++) {
    x[featureIDmap[ids[j]]] = values[j];
  }
  return x;
}

vector<double> SparseDataSet::getFeature(int j)
{
  vector<double> f(size(), 0);
  for (int i = 0; i < size(); ++i) 
	{
    for (featureIterator jiter = X[i].features.begin(); 
		 jiter != X[i].features.end(); ++jiter) {
      if (featureIDmap[(*jiter).index] == j) {
		f[i] = (*jiter).value;
		continue;
      }
    }
  }
  return f;
}

vector<double> SparseDataSet::getPatternValues(int i)
{
    return X[i].getValues();
}

vector<long> SparseDataSet::getPatternKeys(int i)
{
    return X[i].getKeys();
}


void SparseDataSet::getPatternSparse(int i, std::vector<double> &values, std::vector<long> &ids)
{
  X[i].get(values, ids);
}


void SparseDataSet::scale(const std::vector<double> &w)
{
  if (w.size() != numFeatures) 
  {
    cout << "weight vector size " << w.size() << "dim: " << numFeatures << endl;
    printf("wrong size of scaling vector\n");
    //exit(128);  this seemed a bit extreme
    return;
  }
  for (int i = 0; i < n; i++) 
  {
    for (featureIterator j = X[i].features.begin(); 
		 j != X[i].features.end(); ++j)
	  {
		(*j).value *= w[featureIDmap[(*j).index]];
	  }
  }
}

void SparseDataSet::translate(const std::vector<double> &a)
{
  for (int i = 0; i<n; i++) 
  {
    for (featureIterator jiter = X[i].features.begin(); 
		 jiter != X[i].features.end(); ++jiter) {
      (*jiter).value -= a[featureIDmap[(*jiter).index]];
    }
  }
}

std::vector<double> SparseDataSet::mean(const std::vector<int>& patterns)
{
  int p;
  int feature;

  //cout << "computing mean\n";

  //for (int j = 0; j<d; ++j) {
  //  m[j] = 0;
  //}
  // initialization:

//   vector<int> nonzero(d,0);
//   for (int i = 0; i < numPatterns; i++) {
//     p = patterns[i];
//     cout << "i: " << i << endl;
//     for (FeatureListIterator& j = X[p].begin(); 
// 	 j != X[p].end();
// 	 ++j) {
//       feature = featureIDmap[(*j).index];
//       cout << "feat: " << feature << endl;
//       m[feature] += ((*j).value - m[feature]) / float(nonzero[feature] + 1);
//       nonzero[feature] ++;
//     }
//   }
//   for (int j = 0; j < d; ++j) {
//     m[j] *= float(nonzero[j]) / float(n);
//   }

  std::vector<double> m(numFeatures, 0);
  for (unsigned int i = 0; i < patterns.size(); i++) 
	{
  	p = patterns[i];
    //cout << "i: " << i << endl;
    for (featureIterator j = X[p].features.begin();
		 j != X[p].features.end(); ++j) {
	  feature = featureIDmap[(*j).index];
	  //cout << "feat: " << feature << endl;
	  m[feature] += (*j).value;
	}
  }
  for (int j = 0; j < numFeatures; j++) {
    m[j] /= float(n);
  }

  //cout << "done mean\n";
  return m;
}

void SparseDataSet::weightedSum(SparseFeatureVector& w, const std::vector<int> &patterns,
				const std::vector<double> &alpha)
{
  int p;
  vector<double> w_ (numFeatures,0);

  for (unsigned int i = 0; i < patterns.size(); i++) {
    p = patterns[i];
    for (featureIterator jiter = X[p].features.begin();
		 jiter != X[p].features.end(); ++jiter)
	  {
		w_[featureIDmap[(*jiter).index]] += (*jiter).value * alpha[i];
	  }
  }
  w.features.clear();
  w.initialize(featureID, w_);
}

std::vector<double> SparseDataSet::standardDeviation(const std::vector<int>& patterns)
{
  int p;
  int feature;
  vector<int> nonzero(numFeatures,0);
  //vector<double> m(d,0);
  vector<double> sq(numFeatures,0);

  std::vector<double> m = mean(patterns);

  //cout << "computing std\n";
  //mean(m, patterns, numPatterns);
  std::vector<double> s(numFeatures, 0);
  //for (int j = 0; j<d; ++j) {
  //  s[j] = 0;
  //}

  for (unsigned int i = 0; i < patterns.size(); i++) 
	{
    p = patterns[i];
    //for (list<Feature>::iterator j = X[p].begin(); 
    for (featureIterator j = X[p].features.begin();
		 j != X[p].features.end();	 ++j) {
	  feature = featureIDmap[(*j).index];
      sq[feature] += ((*j).value * (*j).value - sq[feature]) / float(nonzero[feature] + 1);
      nonzero[feature] ++;
    }
  }
  for (int j = 0; j < numFeatures; j++) {
    s[j] = sqrt(sq [j] * float(nonzero[j]) / float(n) - m[j] * m[j]);
  }

  return s;

}

std::vector<int> SparseDataSet::featureCounts(const std::vector<int>& patterns)
{
  int p;
  int feature;

  std::vector<int> counts(numFeatures);

  for (unsigned int i = 0; i < patterns.size(); i++) 
	{
    p = patterns[i];
    //cout << "i: " << i << endl;
    for (featureIterator j = X[p].features.begin();
		 j != X[p].features.end(); ++j) {
      feature = featureIDmap[(*j).index];
      if ((*j).value != 0) {
		counts[feature]++;
      }
    }
  }
  return counts;
}

int SparseDataSet::featureCount(int featureToCount, const std::vector<int>& patterns)
{
  int p;
  int count = 0;
  int feature;

  for (unsigned int i = 0; i < patterns.size(); i++) 
	{
    p = patterns[i];
    //cout << "i: " << i << endl;
    for (featureIterator j = X[p].features.begin();
		 j != X[p].features.end(); ++j) {
      feature = featureIDmap[(*j).index];
	  if (feature == featureToCount && ((*j).value != 0.0)) 
		{
		  ++count;
		  break;
		}
    }
  }
  return count;
}


std::vector<int> SparseDataSet::nonzero(int feature, 
										const std::vector<int>& patterns)
{
  int p;
  int f;

  std::vector<int> nz;

  for (unsigned int i = 0; i < patterns.size(); i++) 
	{
    p = patterns[i];
    //cout << "i: " << i << endl;
    for (featureIterator j = X[p].features.begin();
		 j != X[p].features.end(); ++j) {
      f = featureIDmap[(*j).index];
      if (f == feature && ((*j).value != 0.0)) 
			{
				nz.push_back(p);
				break;
			}
    }
  }
  return nz;
}

void SparseDataSet::normalize(int p)
{
  for (int i = 0; i < n; i++){
    double norm = X[i].norm(p);
    if (norm > 0) {
      X[i].scale(1.0/norm);
    }
  }
}

std::vector<int> SparseDataSet::commonFeatures(int pattern1, int pattern2)
{
  return X[pattern1].commonFeatures(X[pattern2]);
}

void SparseDataSet::libsvm_construct(struct svm_problem &prob) 
{
  prob.l = n;
  prob.x = new struct svm_node*[n];
  prob.y = new double[n];

  for (int i = 0; i < n; i++) {
    prob.y[i] = double(Y[i]);
    unsigned int length = X[i].size();
    prob.x[i] = new svm_node[length + 1];
    prob.x[i][length].index = -1;
    prob.x[i][length].value = 0;
    int j = 0;
    for (featureIterator jiter = X[i].features.begin(); 
		 jiter != X[i].features.end(); ++jiter) {
      prob.x[i][j].index = (*jiter).index + 1;
      prob.x[i][j].value = (*jiter).value;
      ++j;
      
    }
  }
}



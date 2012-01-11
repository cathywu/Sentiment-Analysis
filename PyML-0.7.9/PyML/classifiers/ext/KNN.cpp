
# include "KNN.h"

KNN::KNN(int _k = 3) : k(_k)
{
}
    
void KNN::train(DataSet* _data)
{
  data = _data;

  numClasses = 0;
  // xxx use a MAXIMUM function
  for (unsigned int i = 0; i < data->Y.size(); i++){
    if (data->Y[i] > numClasses) {
      numClasses = int(data->Y[i]);
    }
  }
  ++numClasses;

}


std::vector<double> KNN::test(DataSet& testdata)
{
  int p;

  std::vector<int> labels(testdata.size());
  std::vector<double> decisionFunction(testdata.size());
  //decisionFunction.reserve(testdata.size());

  vector<vector<int> > classes(numClasses);      
  for (int i = 0; i < data->size(); i++) {
    classes[int(data->Y[i])].push_back(i);
  }

  for (unsigned int i = 0; i < testdata.size(); i++) {
      //labels.push_back(0);
      vector<double> classSim(numClasses,0);
      for (int c = 0; c < numClasses; c++) {
	  vector<double> similarities(classes[c].size());
	  for (unsigned int j = 0; j < classes[c].size(); j++) {
	      p = classes[c][j];
	      //cout << "about to compute kernel" << endl;
	      similarities[j] = data->kernel->eval(data, p, i, &testdata);
	      //cout << "computed kernel" << endl;
	  }
	  partial_sort(similarities.begin(),
		       similarities.begin() + k,
		       similarities.end(),
		       greater<double>());
	  for (int j = 0; j < k; j++) {
	      classSim[c] += similarities[j];
	  }
      }
      double largestSimilarity = -1e10;
      for (int c = 0; c < numClasses; c++) {
	  if (classSim[c] > largestSimilarity) {
	      largestSimilarity = classSim[c];
	      labels[i] = c;
	  }
      }
      // find the second largest similarity:
      double secondLargestSimilarity = -1e10;
      for (int c = 0; c < numClasses; c++) {
	  if (!(c == labels[i])){
	      if (classSim[c] > secondLargestSimilarity) {
		  secondLargestSimilarity = classSim[c];
	      }
	  }
      }
      decisionFunction[i] = largestSimilarity - secondLargestSimilarity;
      if (numClasses == 2) {
	  decisionFunction[i] = decisionFunction[i] * (labels[i] * 2 - 1);
      }
  }
  cout << labels.size() << " " << decisionFunction.size() << endl;
  for (int i = 0; i<labels.size(); ++i){
      decisionFunction.push_back(labels[i]);
  }
  cout << "done testing KNN*****************************" << endl;

  return decisionFunction;

}

std::vector<int> KNN::nearestNeighbors(DataSet& testdata, int p) 
{
  std::vector<int> neighbors;

  vector<double> similarities(data->size());
  for (int i = 0; i < data->size(); ++i) {
    similarities[i] = data->kernel->eval(data, i, p, &testdata);
  }
  std::vector<double> unorderedSim(similarities);
  partial_sort(similarities.begin(),
	       similarities.begin() + k,
	       similarities.end(),
	       greater<double>());
  for (int j = 0; j < k; ++j) {
    for (int i = 0; i < data->size(); ++i) {
      if (unorderedSim[j] == similarities[i]) {
	neighbors.push_back(i);
      }
    }
  }

  return neighbors;

}



std::vector<double> KNN::classScores(DataSet& testdata, int p)
{
  vector<vector<int> > classes(numClasses);      
  for (int i = 0; i < data->size(); i++) {
    classes[int(data->Y[i])].push_back(i);
  }

  vector<double> classSim(numClasses, 0);
  //cout << "i: " << i << endl;
  for (int c = 0; c < numClasses; c++) {
    vector<double> similarities(classes[c].size());
    for (unsigned int j = 0; j < classes[c].size(); j++) {
      p = classes[c][j];
      similarities[j] = data->kernel->eval(data, classes[c][j], p, &testdata);
    }
    partial_sort(similarities.begin(),
		 similarities.begin() + k,
		 similarities.end(),
		 greater<double>());
    for (int j = 0; j < k; j++) {
      classSim[c] += similarities[j];
    }
  }
  return classSim;
}

int KNN::nearestNeighbor(DataSet& data, int pattern)
{
  double maxSim = -1e10;
  double sim;
  int nn = 0;

  for (int i = 0; i < data.size(); i++) {
    if (i != pattern) {
      sim = data.kernel->eval(&data, pattern, i, &data);
      if (sim > maxSim) {
	maxSim = sim;
	nn = i;

      }
    }
  }
  return nn;
}

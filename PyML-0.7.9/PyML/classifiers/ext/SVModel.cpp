
# include "SVModel.h"

SVModel::SVModel (DataSet &data, 
		  const std::vector<int>& svID, 
		  const std::vector<double>& alpha_, double b_) : 
  svdata(data.duplicate(svID)),
  numSV(svID.size()), 
  alpha(alpha_),
  b(b_) 
{ }

SVModel::~SVModel()
{
  delete svdata;
}

double SVModel::decisionFunc (DataSet *data, int i) {

  double sum = b;
  for (int j = 0; j < svdata->size(); j++) {
    sum += alpha[j] * svdata->kernel->eval(data, i, j, svdata);
  }
  
  return sum;

}

LinearSVModel::LinearSVModel (VectorDataSet &data, 
			      const std::vector<int> &svID, 
			      const std::vector<double>& alpha_, double b_) : 
    b(b_), wVec(data.numFeatures, 0) 
{
    cout << "in Linear SVModel" << endl;
    data.weightedSum(w, svID, alpha_);
    for (long int i = 0; i < w.size(); ++i) {
	wVec[i] = w[i];
    }
    cout << "done Linear SVModel" << endl;
}

LinearSVModel::~LinearSVModel () 
{
}

void LinearSVModel::getW(std::vector<double> &values)
{
    w.get(values);
}
std::vector<double> LinearSVModel::getWvec()
{
    return wVec;
}

double LinearSVModel::decisionFunc (VectorDataSet &data, int i) 
{
    return decisionFunc(data.X[i]);
}

double LinearSVModel::decisionFunc (FeatureVector &x) 
{  
    return w.dotProduct(x) + b;
}

LinearSparseSVModel::LinearSparseSVModel (SparseDataSet &data, 
					  const std::vector<int> &svID, 
					  const std::vector<double>& alpha_, double b_)
  : b(b_), wVec(data.numFeatures, 0)
{

  data.weightedSum(w, svID, alpha_);

  for (featureIterator jiter = w.features.begin();
       jiter != w.features.end();  ++jiter) {
      wVec[data.featureIDmap[(*jiter).index]] = (*jiter).value;
  }
}

LinearSparseSVModel::LinearSparseSVModel(SparseDataSet& data, const std::vector<double>& w_, double b_)
    :b(b_), wVec(w_)
{
    w.initialize(data.getFeatureID(), w_);
}

LinearSparseSVModel::~LinearSparseSVModel () 
{
}

void LinearSparseSVModel::getW(std::vector<double> &values, std::vector<long> &ids)
{
    w.get(values, ids);
}

std::vector<double> LinearSparseSVModel::getWvec()
{
  return wVec;
}

double LinearSparseSVModel::decisionFunc (SparseDataSet &data, int i) 
{
  return decisionFunc(data.X[i]);
}

double LinearSparseSVModel::decisionFunc (SparseFeatureVector &x) 
{  
  return w.dotProduct(x) + b;
}


  
  

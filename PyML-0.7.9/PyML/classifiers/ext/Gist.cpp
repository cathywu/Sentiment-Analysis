# include "Gist.h"
# include <math.h>

Gist::Gist(DataSet *_data, 
	   const std::vector<double> &C_, 
	   const int _cacheSize,
	   const int _maxiter) : 
  data(_data),
  C(C_),
  maxiter(_maxiter),
  eps(0.001), 
  tolerance(0.001),
  convergenceThreshold(1e-4),
  alpha(_data->size()), 
  isLinear(false), 
  Y(_data->size()), 
  cache(_data, _cacheSize)
{
  cout << "constructing gist object" << endl;
  for (int i = 0; i < data->size(); i++) {
    //Kdiag[i] = data->kernel->eval(data, i, i);
    Y[i] = double(data->Y[i]) * 2.0 - 1.0;
  }
  cout << "constructed GIST object" << endl;

}

Gist::~Gist()
{
  cout << "in Gist::~Gist" << endl;
  //delete data;
}

bool Gist::optimize()
{
  int iter = 0;

  while (!converged()) {
    if (iter % 100 == 1) {
      cout << iter << " iterations" << endl;
    }
    vector<int> patterns(size());
    for (int i = 0; i < size(); ++i)
      patterns[i] = i;
    random_shuffle(patterns.begin(), patterns.end());

    for (int i = 0; i < size(); ++i) {
      int pattern = patterns[i];
      alpha[pattern] = updateAlpha(pattern);
    }
    ++iter;
    if ((maxiter != 0) && (iter >=maxiter)) {
      cout << "Warning: svm did not converge after " << iter << endl;
      return false;
    }
	
  }
  return true;
}

double Gist::decisionFunc(int pattern, vector<float> &kernelRow)
{
  double val = 0;
  for (int i = 0; i < size(); ++i) {
    val += alpha[i] * Y[i] * kernelRow[i];
  }
  return val;
}

double Gist::updateAlpha(int pattern)
{
  vector<float> &currentRow = cache.getRow(pattern);
  
  double decisionFuncVal = decisionFunc(pattern, currentRow);
  double newAlpha = 1.0 - Y[pattern] * decisionFuncVal + 
    alpha[pattern] * currentRow[pattern];
  newAlpha /= currentRow[pattern];
  //cout << "new alpha before constraint:" << newAlpha << endl;
  if (newAlpha > C[pattern]) {
    newAlpha = C[pattern];
  }
  else if (newAlpha < 0) {
    newAlpha = 0.0;
  }
  //cout << "new alpha:" << newAlpha << endl;

  return newAlpha;

}

double Gist::objectiveFunction()
{
  double obj = 0.0;

  for (int i = 0; i < size(); ++i) {
    vector<float> &kernelRow = cache.getRow(i);    
    obj += 2.0 * alpha[i] + Y[i] * decisionFunc(i, kernelRow);
  }

  return obj;
}

bool Gist::converged() 
{
  static double prevObjective = 0.0;
  static int iteration = 0;

  iteration++;
  if (iteration == 1) {
    return false;
  }
  double obj = objectiveFunction();
  //cout << "obj: " << obj << endl;
  double delta = obj - prevObjective;
  //cout << "delta: " << delta << endl;
  prevObjective = obj;


  if ((delta < 0.0) && (iteration != 1)) {
    //cout << "Negative delta" << endl;
    return false;
  }
  return (fabs(delta) < convergenceThreshold);
  
}

void Gist::show() {

  cout << "b: " << b << endl;
  cout << "alpha:" << endl;
  for (int i = 0; i < data->size(); i++) {
    cout << alpha[i] << " " << endl;
  }
  cout << endl;

}

void runGist(DataSet *data, 
	     const std::vector<double> &C, 
	     std::vector<double> &alpha,
	     int cacheSize,
	     int iterations)
{

  cout << "running gist" << endl;
  Gist g(data, C, cacheSize, iterations);
  g.optimize();
  alpha = g.alpha;

}


GradientDescent::GradientDescent(DataSet *_data, 
	   const std::vector<double> &C_, 
	   const int _cacheSize,
	   const int _maxiter) : 
  data(_data),
  C(C_),
  maxiter(_maxiter),
  eps(0.001), 
  tolerance(0.001),
  convergenceThreshold(1e-4),
  learningRate(0.1),
  alpha(_data->size()), 
  isLinear(false), 
  Y(_data->size()), 
  cache(_data, _cacheSize)
{
  for (int i = 0; i < data->size(); i++) {
    //Kdiag[i] = data->kernel->eval(data, i, i);
    Y[i] = double(data->Y[i]) * 2.0 - 1.0;
  }
  cout << "constructed GradientDescent object" << endl;

}

GradientDescent::~GradientDescent()
{
  cout << "in GradientDescent::~GradientDescent" << endl;
  //delete data;
}

bool GradientDescent::optimize()
{
  int iter = 0;

  while (!converged()) {
    if (iter % 100 == 1) {
      cout << iter << " iterations" << endl;
    }
    vector<int> patterns(size());
    for (int i = 0; i < size(); ++i)
      patterns[i] = i;
    random_shuffle(patterns.begin(), patterns.end());

    for (int i = 0; i < size(); ++i) {
      int pattern = patterns[i];
      alpha[pattern] = updateAlpha(pattern);
    }
    ++iter;
    if ((maxiter != 0) && (iter >=maxiter)) {
      cout << "Warning: svm did not converge after " << iter << endl;
      return false;
    }
	
  }
  return true;
}

double GradientDescent::decisionFunc(int pattern, vector<float> &kernelRow)
{
  double val = 0;
  for (int i = 0; i < size(); ++i) {
    val += alpha[i] * Y[i] * kernelRow[i];
  }
  return val;
}

double GradientDescent::updateAlpha(int pattern)
{
  vector<float> &currentRow = cache.getRow(pattern);
  double decisionFuncVal = decisionFunc(pattern, currentRow);
  double newAlpha = alpha[pattern] + 
    learningRate * (1 - Y[pattern] * decisionFuncVal);

  if (newAlpha > C[pattern]) {
    newAlpha = C[pattern];
  }
  else if (newAlpha < 0) {
    newAlpha = 0.0;
  }
  //cout << "new alpha:" << newAlpha << endl;

  return newAlpha;

}

double GradientDescent::objectiveFunction()
{
  double obj = 0.0;

  for (int i = 0; i < size(); ++i) {
    vector<float> &kernelRow = cache.getRow(i);    
    obj += 2.0 * alpha[i] + Y[i] * decisionFunc(i, kernelRow);
  }

  return obj;
}

bool GradientDescent::converged() 
{
  static double prevObjective = 0.0;
  static int iteration = 0;

  iteration++;
  if (iteration == 1) {
    return false;
  }
  double obj = objectiveFunction();
  //cout << "obj: " << obj << endl;
  double delta = obj - prevObjective;
  //cout << "delta: " << delta << endl;
  prevObjective = obj;


  if ((delta < 0.0) && (iteration != 1)) {
    //cout << "Negative delta" << endl;
    return false;
  }
  return (fabs(delta) < convergenceThreshold);
  
}

void GradientDescent::show() {

  cout << "b: " << b << endl;
  cout << "alpha:" << endl;
  for (int i = 0; i < data->size(); i++) {
    cout << alpha[i] << " " << endl;
  }
  cout << endl;

}

void runGradientDescent(DataSet *data, 
			const std::vector<double> &C, 
			std::vector<double> &alpha,
			int cacheSize,
			int iterations)
{

  cout << "running gradient descent" << endl;
  GradientDescent g(data, C, cacheSize, iterations);
  g.optimize();
  alpha = g.alpha;

}

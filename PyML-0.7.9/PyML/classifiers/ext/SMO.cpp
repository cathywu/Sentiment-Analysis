# include "SMO.h"

SMO::SMO(DataSet *_data, const std::vector<double> &C_, const int _cacheSize) : 
  data(_data),
  C(C_), 
  eps(0.001), 
  tolerance(0.001), 
  alpha(_data->size()), 
  b(0.0), 
  isLinear(false), 
  Kdiag(_data->size()), 
  Y(_data->size()), 
  G(_data->size(), -1),
  Gbar(_data->size(), 0),
  unshrinked(false),
  shrinking(false),
  cache(_data, _cacheSize)
{
  for (int i = 0; i < data->size(); i++) {
    Kdiag[i] = data->kernel->eval(data, i, i, data);
    Y[i] = double(data->Y[i]) * 2.0 - 1.0;
    //activeSet.insert(i);
    activeSet.push_back(i);
  }

}

SMO::~SMO()
{
  //delete data;
}

void SMO::optimize()
{
  
  int iter = 0;
  int counter = min(data->size(), 1000);

  while (1) {

    if (--counter == 0) {
      counter = min(data->size(), 1000);
      if (shrinking) shrink();
      cout << ".";
    }
    int i,j;
    bool kktViolation = selectWorkingSet(i, j);
    //cout << "kktViolation " << kktViolation << endl;
    //cout << "indices " << i << " " << j << endl;
    if (!kktViolation) { // && !shrinking) {
      break;
    }
    else if (!kktViolation) {
      reconstructGradient();
      for (int k=0; k < size(); k++)
	//activeSet.insert(k);
      cout << "*";
      kktViolation = selectWorkingSet(i, j);
      if (!kktViolation) {
	break;
      }
    }
    ++iter;
    
    update(i, j);
  }

  b = compute_b();
}

void SMO::update(int i, int j) 
{
  double C_i = C[i];
  double C_j = C[j];

  double old_alpha_i = alpha[i];
  double old_alpha_j = alpha[j];
  
  double kii = Kdiag[i];
  double kij = data->kernel->eval(data, i, j, data);
  double kjj = Kdiag[j];

  //cout << "updating... " << endl;

  if(Y[i] != Y[j]) {
    //cout << "Y[i] != Y[j]" << endl;
    double delta = (-G[i]-G[j]) / max((kii + kjj - 2 * kij), 0.0);
    //cout << "delta " << delta << endl;
    double diff = alpha[i] - alpha[j];
    alpha[i] += delta;
    alpha[j] += delta;
			
    if(diff > 0) {
      if(alpha[j] < 0) {
	alpha[j] = 0;
	alpha[i] = diff;
      }
    }
    else {
      if(alpha[i] < 0) {
	alpha[i] = 0;
	alpha[j] = -diff;
      }
    }
    if(diff > C_i - C_j) {
      if(alpha[i] > C_i) {
	alpha[i] = C_i;
	alpha[j] = C_i - diff;
      }
    }
    else {
      if(alpha[j] > C_j) {
	alpha[j] = C_j;
	alpha[i] = C_j + diff;
      }
    }
  }
  else {
    double delta = (G[i]-G[j]) / max((kii + kjj - 2 * kij), 0.0);
    double sum = alpha[i] + alpha[j];
    alpha[i] -= delta;
    alpha[j] += delta;
    if(sum > C_i) {
      if(alpha[i] > C_i) {
	alpha[i] = C_i;
	alpha[j] = sum - C_i;
      }
    }
    else {
      if(alpha[j] < 0) {
	alpha[j] = 0;
	alpha[i] = sum;
      }
    }
    if(sum > C_j) {
      if(alpha[j] > C_j) {
	alpha[j] = C_j;
	alpha[i] = sum - C_j;
      }
    }
    else {
      if(alpha[i] < 0) {
	alpha[i] = 0;
	alpha[j] = sum;
      }
    }
  }

  //cout << "alpha[i] " << alpha[i] << endl;
  //cout << "alpha[j] " << alpha[j] << endl;

  // update G

  double delta_alpha_i = alpha[i] - old_alpha_i;
  double delta_alpha_j = alpha[j] - old_alpha_j;

  int k;
  vector<float> &irow = cache.getRow(i);
  vector<float> &jrow = cache.getRow(j);

  for (IndexSetItr itr=activeSet.begin(); itr!=activeSet.end(); ++itr) {
    k = (*itr);
    //G[k] += data->kernel->eval(data->X[i], data->X[k]) * delta_alpha_i * Y[i] * Y[k] + 
    //data->kernel->eval(data->X[j], data->X[k]) * Y[j] * Y[k] * delta_alpha_j;    
    G[k] += irow[k] * delta_alpha_i * Y[i] * Y[k] + 
      jrow[k] * Y[j] * Y[k] * delta_alpha_j;
    
  }

  // update G_bar
  
}

void SMO::reconstructGradient()
{
  // reconstruct inactive elements of G from G_bar and free variables
  if (activeSet.size() == data->size()) return;

  int i;
  for(IndexSetItr itr=activeSet.begin(); itr!=activeSet.end(); ++itr) {
    i = (*itr);
    G[i] = Gbar[i] + 1.0; /// 1 instead of b[i];
  }

  for(IndexSetItr itr=activeSet.begin(); itr!=activeSet.end(); ++itr) {
    i = *itr;
    if (isFree(i)) {
      vector<float> &irow = cache.getRow(i);
      double alpha_i = alpha[i];
      for (int j = 0; j < size(); j++) {
	//if (activeSet.find(j) != activeSet.end()) 
	//  continue;
	/// there is some waste here since we don't need all the entries;
	/// does not matter yet since we are not doing shrinking...
	G[j] += alpha_i * Y[i] * Y[j] * irow[j];
      }
    }
  }
}

bool SMO::selectWorkingSet(int &iOut, int &jOut)
{

  double Gmax1 = -INF;		// max { -grad(f)_i * d | y_i*d = +1 }
  int Gmax1_idx = -1;
  
  double Gmax2 = -INF;		// max { -grad(f)_i * d | y_i*d = -1 }
  int Gmax2_idx = -1;

  //cout << "in select working set" << endl;
  int i;
  for(IndexSetItr itr = activeSet.begin(); itr != activeSet.end(); ++itr){
    i = *itr;
    if(Y[i]==1) {
      //cout << "Y[i] " << Y[i] << endl;
      if(!isUpperBound(i)) {	// d = +1
	//cout << "not upper 1 " << G[i] << endl;
	if(-G[i] > Gmax1) {
	  Gmax1 = -G[i];
	  Gmax1_idx = i;
	}
      }
      if(!isLowerBound(i)) {	// d = -1
	if(G[i] > Gmax2) {
	  Gmax2 = G[i];
	  Gmax2_idx = i;
	}
      }
    }
    else {  // y = -1
      //cout << "Y[i] " << Y[i] << endl;
      if(!isUpperBound(i)) { // d = +1
	//cout << "not upper -1 " << G[i] << endl;
	if(-G[i] > Gmax2) {
	  Gmax2 = -G[i];
	  Gmax2_idx = i;
	}
      }
      if(!isLowerBound(i)) {	// d = -1
	if(G[i] > Gmax1) {
	  Gmax1 = G[i];
	  Gmax1_idx = i;
	}
      }
    }
  }
  //KKT conditions are satisfied:
  if(Gmax1+Gmax2 < eps) {
    return false;
  }

  //return the pair that has the largest KKT violation:
  iOut = Gmax1_idx;
  jOut = Gmax2_idx;

  //cout << " iOut " << iOut << " jOut " << jOut << endl;

  return true;
}

void SMO::shrink()
{
}

double SMO::compute_b()
{
  double r;
  int nr_free = 0;
  double ub = INF, lb = -INF, sum_free = 0;
  
  int i;
  for(IndexSetItr itr=activeSet.begin(); itr!=activeSet.end(); ++itr) {
    i = *itr;
    double yG = Y[i]*G[i];
    if (isLowerBound(i)) {
      if(Y[i] > 0)
	ub = min(ub,yG);
      else
	lb = max(lb,yG);
    }
    else if (isUpperBound(i)) {
      if(Y[i] < 0)
	ub = min(ub,yG);
      else
	lb = max(lb,yG);
    }
    else {
      ++nr_free;
      sum_free += yG;
    }
  }
  if(nr_free>0) {
    r = sum_free / double(nr_free);
  }
  else
    r = (ub+lb)/2;
  
  return r;

}

void SMO::show() {

  cout << "b: " << b << endl;
  cout << "alpha:" << endl;
  for (int i = 0; i < data->size(); i++) {
    cout << alpha[i] << " " << endl;
  }
  cout << endl;

}

std::vector<double> runSMO(DataSet *data, 
			   const std::vector<double> &C, 
			   //	      std::vector<double> &alpha,
			   int cacheSize)
{

  SMO s(data, C, cacheSize);
  s.optimize();
  //alpha = s.alpha;
  s.alpha.push_back(s.b);
  return s.alpha;
}

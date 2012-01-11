void SMO::shrink()
{
  int i,j;
  if (!selectWorkingSet(i,j)) return;

  double Gm1 = -Y[j]*G[j];
  double Gm2 = Y[i]*G[i];

  // shrink
  vector<int> activesToErase;
  for(IndexSetItr itr=activeSet.begin(); itr!=activeSet.end(); ++itr){
    int k = *itr;
    if (isLowerBound(k)) {
      if (Y[k] == +1) {
	if (-G[k] >= Gm1) continue;
      }
      else {
	if (-G[k] >= Gm2) continue;
      }
    }
    else if (isUpperBound(k)) {
      if (Y[k] == +1) {
	if (G[k] >= Gm2) continue;
      }
      else {
	if(G[k] >= Gm1) continue;
      }
    }
    else continue;
    activesToErase.push_back(k);
  }
  for (int k = 0; k < activesToErase.size(); ++k) {
    activeSet.erase(activesToErase[k]);
  }

  // unshrink, check all variables again before final iterations

  if(unshrinked || -(Gm1 + Gm2) > eps*10) return;
	
  unshrinked = true;
  reconstructGradient();

  vector<int> addToActive;
  for(int k=0; k < size(); ++k) {
    if (activeSet.find(k) != activeSet.end()) continue;
    if(isLowerBound(k)) {
      if(Y[k]==+1) {
	if(-G[k] < Gm1) continue;
      }
      else {	
	if(-G[k] < Gm2) continue;
      }
    }
    else if(isUpperBound(k)) {
      if(Y[k] == +1) {
	if(G[k] < Gm2) continue;
      }
      else {
	if(G[k] < Gm1) continue;
      }
    }
    else continue;
    addToActive.push_back(k);
  }
  for (int k = 0; k < addToActive.size(); ++k) {
    cout << "restored " << k << endl;
    activeSet.insert(addToActive[k]);
  }
  cout << "size of active: " << activeSet.size() << endl;
  
}


  {
    bool upperI = isUpperBound(i);
    bool upperJ = isUpperBound(j);
    bool wasUpperI = (old_alpha_i >= C[i]);
    bool wasUpperJ = (old_alpha_j >= C[j]);

    if (upperI != wasUpperI) {
      if (wasUpperI)
	for(k=0; k<size(); ++k)
	  //Gbar[k] -= C_i * Y[i] * Y[k] * data->kernel->eval(data->X[i], data->X[k]);
	  Gbar[k] -= C_i * Y[i] * Y[k] * irow[k];
      else
	for(k=0; k<size(); ++k)
	  //Gbar[k] += C_i * Y[i] * Y[k] * data->kernel->eval(data->X[i], data->X[k]);
	  Gbar[k] += C_i * Y[i] * Y[k] * irow[k];
    }
    if (upperJ != wasUpperJ) {
      if (wasUpperJ)
	for(k=0; k<size(); ++k)
	  //Gbar[k] -= C_j * Y[j] * Y[k] * data->kernel->eval(data->X[j], data->X[k]);
	  Gbar[k] -= C_j * Y[j] * Y[k] * jrow[k];
      else
	for(k=0; k<size(); ++k)
	  //Gbar[k] += C_j * Y[j] * Y[k] * data->kernel->eval(data->X[j], data->X[k]);
	  Gbar[k] += C_j * Y[j] * Y[k] * jrow[k];
    }
  }

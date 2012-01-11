# include "Aggregate.h"

Aggregate::Aggregate(int n) : DataSet(n), ownData(false)
{ 
  //cout << "constructed Aggregate" << endl;
}

Aggregate::Aggregate(int n, const std::vector<double> &weights_) : 
  DataSet(n), ownData(false), weights(weights_)
{ 
  //cout << "constructed Aggregate" << endl;
}

Aggregate::~Aggregate() 
{
  if (ownData) {
    for (unsigned int i = 0; i < datas.size(); ++i) {
      delete datas[i];
    }
  }
  //cout << "in Aggregate::~Aggregate" << endl;
}

Aggregate::Aggregate(const Aggregate &other, const std::vector<int> &patterns) : 
  DataSet(other, patterns), weights(other.weights), ownData(true)
{
  for (unsigned int i = 0; i < other.datas.size(); ++i) {
    datas.push_back(other.datas[i]->duplicate(patterns));
  }
  //cout << "done copy\n";
  
}

double Aggregate::dotProduct(int i, int j)
{
  Aggregate *other;
  other = this;
  return dotProduct(i, j, other);
}

double Aggregate::dotProduct(int i, int j, DataSet *other_)
{
  Aggregate *other;
  if (other_ == 0) 
    other = this;
  else 
    other = dynamic_cast<Aggregate *>(other_);

  double sum = 0;
  for (unsigned int idx = 0; idx < datas.size(); ++idx) {
    //sum += datas[idx]->dotProduct(i, j, other->datas[idx]);
    sum += weights[idx] * datas[idx]->kernel->eval(datas[idx], i, j, other->datas[idx]);
  }
  return sum;
}

void Aggregate::show()
{

}



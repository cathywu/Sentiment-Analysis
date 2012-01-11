
# include "PairDataSet.h"

PairDataSet::PairDataSet(const std::vector<int>& first_, 
			 const std::vector<int>& second_, 
			 DataSet *data_) :
  DataSet(first_.size()),
  first(first_), second(second_), data(data_)
{ 
  
  cout << "constructed PairDataSet" << endl;
}

PairDataSet::~PairDataSet() 
{
  cout << "in PairDataSet::~PairDataSet" << endl;
}

PairDataSet::PairDataSet(const PairDataSet &other, 
			 const std::vector<int> &patterns) : 
  DataSet(other, patterns),
  data(other.data)
{
  for (unsigned int i = 0; i < patterns.size(); i++) {
    int p = patterns[i];
    Y[i] = other.Y[p];
    first.push_back(other.first[p]);
    second.push_back(other.second[p]);
  }
  cout << "done copy\n";
  
}

double PairDataSet::dotProduct(int i, int j)
{
  PairDataSet *other;
  other = this;

  return dotProduct(i, j, other);
}

double PairDataSet::dotProduct(int i, int j, DataSet *other_)
{
  PairDataSet *other;
  other = dynamic_cast<PairDataSet *>(other_);

  return 
    this->data->kernel->eval(this->data, first[i], other->first[j], other->data) *
    other->data->kernel->eval(this->data, second[i], other->second[j], other->data)+
    this->data->kernel->eval(this->data, first[i], other->second[j], other->data) *
    other->data->kernel->eval(this->data, second[i], other->first[j], other->data);
}

void PairDataSet::show()
{
  for (int i = 0 ; i < 6; i++) {
    cout << first[i] << " " << second[i] << " " << Y[i] << endl;
  }
}






PairDataSetSum::PairDataSetSum(const std::vector<int>& first_, 
			       const std::vector<int>& second_, 
			       DataSet *data_) :
  PairDataSet(first_, second_, data_)
{ 
  cout << "constructed PairDataSetSum" << endl;
}

PairDataSetSum::~PairDataSetSum() 
{
  cout << "in PairDataSetSum::~PairDataSetSum" << endl;
}

PairDataSetSum::PairDataSetSum(const PairDataSetSum &other, 
			       const std::vector<int> &patterns) : 
  PairDataSet(other, patterns)
{
  cout << "done copy\n";
}

double PairDataSetSum::dotProduct(int i, int j)
{
  PairDataSetSum *other;
  other = this;

  return dotProduct(i, j, other);
}
			 
double PairDataSetSum::dotProduct(int i, int j, DataSet *other_)
{
  PairDataSetSum *other;
  if (other_ == 0) 
    other = this;
  else 
    other = dynamic_cast<PairDataSetSum *>(other_);

  //   double K11 = this->data->kernel->eval(this->data, first[i], 
  // 					other->first[j], other->data);
  
  //   double K22 = this->data->kernel->eval(this->data, second[i], 
  // 					other->second[j], other->data);
  //   double K12 = this->data->kernel->eval(this->data, first[i], 
  // 					other->second[j], other->data);
  
  //   double K21 = this->data->kernel->eval(this->data, second[i], 
  // 					other->first[j], other->data);

  //   if ((K11 >= K22 && K11 >= K12 && K11 >= K21) ||
  //       (K22 >= K11 && K22 >= K12 && K22 >= K21)) {
  //     return K11 * K22;
  //   }
  //   else {
  //     return K12 * K21;
  //   }

  double K11 = this->data->kernel->eval(this->data, first[i], 
					other->first[j], other->data);
  
  double K22 = this->data->kernel->eval(this->data, second[i], 
   					other->second[j], other->data);
  double K12 = this->data->kernel->eval(this->data, first[i], 
   					other->second[j], other->data);
  
  double K21 = this->data->kernel->eval(this->data, second[i], 
   					other->first[j], other->data);

  return K11 + K22 + K12 + K21;
  
}



PairDataSetOrd::PairDataSetOrd(const std::vector<int>& first_, 
			       const std::vector<int>& second_, 
			       DataSet *data_) :
  PairDataSet(first_, second_, data_)
{ 
  cout << "constructed PairDataSetOrd" << endl;
}

PairDataSetOrd::~PairDataSetOrd() 
{
  cout << "in PairDataSetOrd::~PairDataSetOrd" << endl;
}

PairDataSetOrd::PairDataSetOrd(const PairDataSetOrd &other, 
			       const std::vector<int> &patterns) : 
  PairDataSet(other, patterns)
{
  cout << "done copy\n";
}

double PairDataSetOrd::dotProduct(int i, int j)
{
  PairDataSetOrd *other;
  other = this;

  return dotProduct(i, j, other);
}

double PairDataSetOrd::dotProduct(int i, int j, DataSet *other_)
{
  PairDataSetOrd *other;
  if (other_ == 0) 
    other = this;
  else 
    other = dynamic_cast<PairDataSetOrd *>(other_);

  double K11 = this->data->kernel->eval(this->data, first[i], 
					other->first[j], other->data);
  
  double K22 = this->data->kernel->eval(this->data, second[i], 
   					other->second[j], other->data);

  return K11 + K22;
  
}

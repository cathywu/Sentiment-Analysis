
# include "SetData.h"

SetData::SetData(int size, DataSet *data_) :
  DataSet(size)
{ 
    data = data_;
    cout << "constructed SetData" << endl;
}

SetData::~SetData() 
{
  cout << "in SetData::~SetData" << endl;
}

SetData::SetData(const SetData &other, 
		 const std::vector<int> &patterns) : 
  DataSet(other, patterns),
  data(other.data)
{
  for (unsigned int i = 0; i < patterns.size(); i++) {
    int p = patterns[i];
    Y[i] = other.Y[p];
    sets.push_back(other.sets[p]);
  }
  cout << "done copy\n";
  
}

double SetData::dotProduct(int i, int j)
{
  SetData *other;
  other = this;

  return dotProduct(i, j, other);
}

double SetData::dotProduct(int i, int j, DataSet *other_)
{
  SetData *other;
  other = dynamic_cast<SetData *>(other_);
  
  float value = 0;
  for (int k = 0; k < sets[i].size(); ++k) {
      for (int l = 0; l < other->sets[j].size(); ++l) {
	  value += data->kernel->eval(this->data, sets[i][k], other->sets[j][l], other->data);
      }
  }
  return value / (sets[i].size() * other->sets[j].size());

}

void SetData::show()
{
    for (int i = 0 ; i < 2; i++) {
	cout << sets[i][0] << endl;
    }
}





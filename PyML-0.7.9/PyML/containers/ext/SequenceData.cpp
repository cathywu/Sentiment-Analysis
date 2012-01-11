#include "SequenceData.h"
#include "DataSet.h"

# include <algorithm>

# define max(a,b) ((a) > (b) ? (a) : (b))
# define min(a,b) ((a) > (b) ? (b) : (a))
# define max3(a,b,c) max(a, max(b, c))
# define min3(a,b,c) min(a, min(b, c))
# define abs(a) ((a) >= (0) ? (a) : (-a))

SequenceData::SequenceData() 
{ }

SequenceData::SequenceData(const int numPatterns) :
    DataSet(numPatterns), shiftWeight(1, 0)
{ }

SequenceData::SequenceData(const SequenceData &other, const std::vector<int> &patterns)
    : DataSet(other, patterns),
      mink(other.mink), 
      maxk(other.maxk),
      mismatches(other.mismatches),
      mismatchProfile(other.mismatchProfile),
      maxShift(other.maxShift),
      noShiftStart(other.noShiftStart),
      noShiftEnd(other.noShiftEnd),
      shiftWeight(other.shiftWeight)
{

  X.reserve(patterns.size());
  for (unsigned int i = 0; i < patterns.size(); ++i){
    int p = patterns[i];
    X.push_back(other.X[p]);
    Y[i] = other.Y[p];
  }

}

SequenceData::~SequenceData()
{ }

void SequenceData::addPattern(const std::string &sequence)
{
  X.push_back(sequence);
}

void SequenceData::show()
{
    cout << "mink " << mink << endl;
    cout << "maxk " << maxk << endl;
    cout << "mismatches " << mismatches << endl;
    cout << "mismatch profile ";
    for (int i =0; i < mismatchProfile.size(); ++i) {
	cout << " " << mismatchProfile[i];
    }
    cout << endl;
    for (int i =0; i < shiftWeight.size(); ++i) {
	cout << " " << shiftWeight[i];
    }

    cout << endl;
    cout << "max shift " << maxShift << endl;
    cout << "no shift start " << noShiftStart << endl;
    cout << "no shift end " << noShiftEnd << endl;

  cout << "size : " << size() << endl;
  for( int i = 0; i < size(); i++){
	cout << i << " " << X[i] << endl;
  }
}

int SequenceData::shiftSize(const int pos, const int length)
{
    if (pos >= noShiftStart && pos < noShiftEnd) {
	return 0;
    }
    return min3(maxShift, pos, length - (pos + maxk) );
}

double SequenceData::dotProduct(int i, int j, DataSet* dataj)
{
    SequenceData *data2;
    //data1 = dynamic_cast<PositionalKmerData *>(datai);
    data2 = dynamic_cast<SequenceData *>(dataj);
    std::string x = X[i];
    std::string y = data2->X[j];

    //if (x.size() != y.size()) {
    //return 0.0;
    //}
    double value = 0;
    // loop over kmer position
    for (int l = 0; l < min(x.size(), y.size()) - mink + 1; ++l) {
	//cout << "l: " << l << endl;
	int maximumShift = shiftSize(l, x.size());
	//cout << "maximumShift " << maximumShift << endl;
	for (int s = -maximumShift; s <= maximumShift; ++s) {
	    //cout << "shift: " << s << endl;
	    if (l + s < 0) continue;
	    // maximum kmer length depends on where we are in the sequence:
	    int local_maxk = min3(maxk, x.size() - l, y.size() - (l + s) );
	    //cout << "local_maxk " << local_maxk << endl;
	    int m = 0;
	    // loop over kmer length
	    for (int k = 0; k < local_maxk; ++k) {
		//cout << "k: " << k << endl;
		// ignore positions with gap symbols
		//if ((x[l + k] == '-') || (x[l + k] == '.') || (x[l +k] == '_') ||
		//    (y[l + s + k] == '-') || (y[l + s + k] == '.') || (y[l + s + k] == '_')) {
		//    m++;
		//}
		//else {
		    if (x[l + k] != y[l + s + k]){
			m++;
		    }
		    //}
		if (m > mismatchProfile[maxk - 1]) break;
		if (m > mismatchProfile[k] && k >= mink - 1) break;
		
		if (k >= mink - 1) {
		    value += shiftWeight[abs(s)];
		}
	    }
	}
    }

    return value;
}


# include "StringKernel.h"
# include <algorithm>

# define max(a,b) ((a) > (b) ? (a) : (b))
# define min(a,b) ((a) > (b) ? (b) : (a))
# define max3(a,b,c) max(a, max(b, c))
# define min3(a,b,c) min(a, min(b, c))
# define abs(a) ((a) >= (0) ? (a) : (-a))

StringKernel::StringKernel()
{ }

StringKernel::~StringKernel()
{ }

PositionalKmer::PositionalKmer(const int mink_, 
			       const int maxk_, 
			       const int mismatches_,
			       const std::vector<int>& mismatchProfile_,
			       const int maxShift_,
			       const int noShiftStart_,
			       const int noShiftEnd_) :
  mink(mink_), 
  maxk(maxk_), 
  mismatches(mismatches_),
  mismatchProfile(mismatchProfile_),
  maxShift(maxShift_),
  noShiftStart(noShiftStart_),
  noShiftEnd(noShiftEnd_),
  shiftWeight(maxShift_ + 1, 0)
{ 
  for (int s = 0; s <= maxShift; ++s){
	shiftWeight[s] = 1.0 / (2 * (abs(s) + 1));
  }
}

PositionalKmer::PositionalKmer(const PositionalKmer &other) :
  mink(other.mink), 
  maxk(other.maxk),
  mismatches(other.mismatches),
  mismatchProfile(other.mismatchProfile),
  maxShift(other.maxShift),
  noShiftStart(other.noShiftStart),
  noShiftEnd(other.noShiftEnd),
  shiftWeight(other.shiftWeight)
{ }

PositionalKmer::~PositionalKmer()
{ }

int PositionalKmer::shiftSize(const int pos, const int length)
{
  if (pos >= noShiftStart && pos < noShiftEnd) {
    return 0;
  }

  return min3(maxShift, pos, length - (pos + maxk) );
}

double PositionalKmer::eval(const std::string &x, const std::string &y, int i, int j)
{
    double value = 0;
    // loop over kmer position
    for (int l = 0; l < x.size() - mink + 1; ++l) {
	//cout << "l: " << l << endl;
	int maximumShift = shiftSize(l, x.size());
	//cout << "maximumShift " << maximumShift << endl;
	for (int s = -maximumShift; s <= maximumShift; ++s) {
	    //cout << "shift: " << s << endl;
	    if (l + s < 0) continue;
	    // maximum kmer length depends on where we are in the sequence:
	    int maxk = min3(maxk, x.size() - l, y.size() - (l + s) );
	    int m = 0;
	    // loop over kmer length
	    for (int k = 0; k < maxk; ++k) {
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


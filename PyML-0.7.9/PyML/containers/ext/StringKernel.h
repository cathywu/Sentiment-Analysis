# ifndef STRINGKERNEL_H
# define STRINGKERNEL_H

# include <vector>
# include <string>
# include <iostream>
# include <fstream>

using namespace std;

class StringKernel {
 public:

  StringKernel();
  virtual ~StringKernel();
  
  virtual StringKernel* duplicate() = 0;   //"virtual" copy constructor

  virtual StringKernel* castToBase() = 0;

  virtual double eval(const std::string& x, const std::string& y, int i, int j) = 0;

};

class PositionalKmer : public StringKernel
{
 public :

  PositionalKmer(const int mink_, const int maxk_, 
		 const int mismatches_, const std::vector<int>& mismatchProfile_,
		 const int maxShift_, 
		 const int noShiftStart_, const int noShiftEnd_);
				 
  PositionalKmer(const PositionalKmer &other);
  ~PositionalKmer();

  int mink;
  int maxk;

  //the maximum allowed shift:
  int maxShift;

  int noShiftStart;
  int noShiftEnd;

  int mismatches;
  std::vector<int> mismatchProfile;

  std::vector<double> shiftWeight;

  void setMismatchProfile(const std::vector<int>& profile) { mismatchProfile = profile; }

  virtual PositionalKmer* duplicate()
    {return new PositionalKmer(*this); }

  StringKernel* castToBase() { return dynamic_cast<StringKernel *>(this); }

  double eval(const std::string& x, const std::string& y, int i, int j);

  // compute the amount of shift at a particular position
  int shiftSize(const int pos, const int length);

};

# endif

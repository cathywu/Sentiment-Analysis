//FeatureVectorIterator.h
//contains several inline functions for optimization
#ifndef FeatureVectorIterator_H
#define FeatureVectorIterator_H

//#include "Feature.cpp"
struct Feature;//just so it knows
#include "BaseIterator.h"
# include <vector>
#include <list>
/* FeatureVectorIterator is a shell for the STL vector::iterator
This shell is necessary to link list and vector objects and iterators together
by inheritance.  BaseIterator assures that we have all the same functions.
*/


using namespace std;

class FeatureVectorIterator : public BaseIterator
{
	public:
		vectorIterator target;
		
		FeatureVectorIterator(vectorIterator it) {target = it;}
		inline FeatureVectorIterator& operator ++ ()	{target++; 	return *this;}
		inline double operator * ()	{return *target;}
		inline double operator [](int i) {return target[i];}
		inline FeatureVectorIterator& operator + (int i) {target += i;  return *this;}
		inline bool operator != (BaseIterator& other) 
		{
			FeatureVectorIterator* ptr = dynamic_cast<FeatureVectorIterator*>(&other);
			//error message
			return target != ptr->target;
		}
			

	private:
};

#endif

//FeatureListIterator.h
//contains several inline functions for optimization
#ifndef FeatureListIterator_H
#define FeatureListIterator_H

//#include "Feature.cpp"
#include "BaseIterator.h"
# include <vector>
#include <list>
/* FeatureListIterator is basically a switch box so that STL
list and vector can be treated the same.  It delegates all actual 
responsibility to the STL iterators.  
*/


using namespace std;
typedef list<Feature>::iterator listIterator;

class FeatureListIterator : public BaseIterator {
	public:
		listIterator target;

		FeatureListIterator(listIterator it) {	target = it;};
		inline FeatureListIterator& operator ++ (){target++; 	return *this;}
		inline Feature& operator * ()	{return *target;}
		inline bool operator != (BaseIterator& other) 
		{
			FeatureListIterator* ptr = dynamic_cast<FeatureListIterator*>(&other);
			//error message
			return target != ptr->target;
		}
		

	private:
};

#endif

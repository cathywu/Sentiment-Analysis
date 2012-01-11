# ifndef FEATURE_H
# define FEATURE_H

struct Feature {
  long index;
  double value;

	Feature::Feature(long featureID, double featureValue)
	{
  	index = featureID;
  	value = featureValue;
	}

};

# endif


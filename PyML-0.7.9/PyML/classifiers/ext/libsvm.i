%module libsvm
%{
#include "libsvm.h"
%}

enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

%include "std_vector.i"

class DecisionFunction
{
 public:
  int numSV;
  int numBSV;
  std::vector<int> svID;
  std::vector<double> alpha;
  double rho;

  DecisionFunction();
  ~DecisionFunction();
};


struct svm_node
{
	int index;
	double value;
};

struct svm_problem
{
	int l;
	double *y;
	struct svm_node **x;
};

struct svm_parameter
{
	int svm_type;
	int kernel_type;
	int degree;	/* for poly */
	double gamma;	/* for poly/rbf/sigmoid */
	double coef0;	/* for poly/sigmoid */

	/* these are for training only */
	double cache_size; /* in MB */
	double eps;	/* stopping criteria */
	double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
	int nr_weight;		/* for C_SVC */
	int *weight_label;	/* for C_SVC */
	double* weight;		/* for C_SVC */
	double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
	double p;	/* for EPSILON_SVR */
	int shrinking;	/* use the shrinking heuristics */
	int probability; /* do probability estimates */
};

%nodefaultctor;
%nodefaultdtor;



void svm_train_one_pyml(const struct svm_problem *prob, const struct svm_parameter *param,
						double Cp, double Cn, DecisionFunction &f);

%include carrays.i
%array_class(int, intArray);
%array_class(double, doubleArray);



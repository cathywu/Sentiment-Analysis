#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdarg.h>
#include "mylinear.h"
//#include "tron.h"


typedef signed char schar;
//template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
//#ifndef min
//template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
//#endif
//#ifndef max
//template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
//#endif
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{   
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))
#define INF HUGE_VAL

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}

static void (*liblinear_print_string) (const char *) = &print_string_stdout;

#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*liblinear_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif


// A coordinate descent algorithm for 
// L1-loss and L2-loss SVM dual problems
//
//  min_\alpha  0.5(\alpha^T (Q + D)\alpha) - e^T \alpha,
//    s.t.      0 <= alpha_i <= upper_bound_i,
// 
//  where Qij = yi yj xi^T xj and
//  D is a diagonal matrix 
//
// In L1-SVM case:
// 		upper_bound_i = Cp if y_i = 1
// 		upper_bound_i = Cn if y_i = -1
// 		D_ii = 0
// In L2-SVM case:
// 		upper_bound_i = INF
// 		D_ii = 1/(2*Cp)	if y_i = 1
// 		D_ii = 1/(2*Cn)	if y_i = -1
//
// Given: 
// x, y, Cp, Cn
// eps is the stopping tolerance
//
// solution will be put in w
// 
// See Algorithm 3 of Hsieh et al., ICML 2008

#undef GETI
#define GETI(i) (y[i]+1)
// To support weights for instances, use GETI(i) (i)

std::vector<double> solve_l2r_l1l2_svc(SparseDataSet *data, double eps, 
				       double Cp, double Cn, int solver_type)
{
    int l = data->size();
    int w_size = data->numFeatures;

    std::vector<double> w;
    int i, s, iter = 0;
    double C, d, G;
    double *QD = new double[l];
    int max_iter = 1000;
    int *index = new int[l];
    double *alpha = new double[l];
    schar *y = new schar[l];
    int active_size = l;

	// PG: projected gradient, for shrinking and stopping
	double PG;
	double PGmax_old = INF;
	double PGmin_old = -INF;
	double PGmax_new, PGmin_new;

	// default solver_type: L2R_L2LOSS_SVC_DUAL
	double diag[3] = {0.5/Cn, 0, 0.5/Cp};
	double upper_bound[3] = {INF, 0, INF};
	if(solver_type == L2R_L1LOSS_SVC_DUAL)
	{
	    info("solver type %d\n", solver_type);
		diag[0] = 0;
		diag[2] = 0;
		upper_bound[0] = Cn;
		upper_bound[2] = Cp;
	}

	for(i=0; i<w_size; i++)
	    w.push_back(0.0);
	for(i=0; i<l; i++)
	{
		alpha[i] = 0.0;
		if(data->Y[i] > 0)
		{
			y[i] = +1; 
		}
		else
		{
			y[i] = -1;
		}
		QD[i] = diag[GETI(i)] + data->dotProduct(i, i);
		//info("QD %f\n", QD[i]);
		//feature_node *xi = prob->x[i];
		//while (xi->index != -1)
		//{
		//	QD[i] += (xi->value)*(xi->value);
		//	xi++;
		//}
		index[i] = i;
	}

	while (iter < max_iter)
	{
		PGmax_new = -INF;
		PGmin_new = INF;

		for (i=0; i<active_size; i++)
		{
			int j = i+rand()%(active_size-i);
			//info("rand index %d\n",j);
			int temp = index[i];
			index[i] = index[j];
			index[j] = temp;
		}

		for (s=0; s<active_size; s++)
		{
			i = index[s];
			G = 0;
			schar yi = y[i];
			//info("yi %d\n", y[i]);
			for (featureIterator xj = data->X[i].features.begin();
			     xj != data->X[i].features.end(); ++xj)
			    {
				//info("index %d\n", (*xj).index);
				//info("map %d\n", data->featureIDmap[(*xj).index]);
				//info("value %f\n", (*xj).value);
				G += w[data->featureIDmap[(*xj).index]] * (*xj).value ;
			    }
			//info("%f\n", G);
			//feature_vector xi = prob->x[i];
			//while(xi->index!= -1)
			//{
			//	G += w[xi->index-1]*(xi->value);
			//	xi++;
			//}
			G = G*yi-1;

			C = upper_bound[GETI(i)];
			G += alpha[i]*diag[GETI(i)];

			PG = 0;
			if (alpha[i] == 0)
			{
				if (G > PGmax_old)
				{
				    active_size--;
				    int temp = index[s];
				    index[s] = index[active_size];
				    index[active_size] = temp;

				    //swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G < 0)
					PG = G;
			}
			else if (alpha[i] == C)
			{
				if (G < PGmin_old)
				{
					active_size--;
					int temp = index[s];
					index[s] = index[active_size];
					index[active_size] = temp;
					//swap(index[s], index[active_size]);
					s--;
					continue;
				}
				else if (G > 0)
					PG = G;
			}
			else
				PG = G;

			if (PG > PGmax_new) {
			    PGmax_new = PG;
			}
			if (PG< PGmin_new) {
			    PGmin_new = PG;
			}
			//info("PG: %f\n", PG);
			if(fabs(PG) > 1.0e-12)
			{
				double alpha_old = alpha[i];
				alpha[i] = alpha[i] - G/QD[i];
				//info("alpha: %f\n", alpha[i]);
				//info("alpha_old: %f\n", alpha_old);
				if (alpha[i] < 0) {
				    alpha[i] = 0;
				}
				if (alpha[i] > C) {
				    alpha[i] = C;
				}
				//info("alpha: %f\n", alpha[i]);
				//alpha[i] = min(max(alpha[i] - G/QD[i], 0.0), C);
				//info("yi %d\n", yi);
				d = (alpha[i] - alpha_old)*yi;
				//info("d %f\n", d);
				for (featureIterator xi = data->X[i].features.begin();
				     xi != data->X[i].features.end(); ++xi)
				    {
					w[data->featureIDmap[(*xi).index]] += d * (*xi).value;
				    }
				
				//xi = prob->x[i];
				//while (xi->index != -1)
				//{
				//	w[xi->index-1] += d*xi->value;
				//	xi++;
				//}
			}
		}

		iter++;
		if(iter % 10 == 0)
			info(".");

		if(PGmax_new - PGmin_new <= eps)
		{
			if(active_size == l)
				break;
			else
			{
				active_size = l;
				info("*");
				PGmax_old = INF;
				PGmin_old = -INF;
				continue;
			}
		}
		PGmax_old = PGmax_new;
		PGmin_old = PGmin_new;
		if (PGmax_old <= 0)
			PGmax_old = INF;
		if (PGmin_old >= 0)
			PGmin_old = -INF;
	}

	info("\noptimization finished, #iter = %d\n",iter);
	if (iter >= max_iter)
		info("\nWARNING: reaching max number of iterations\nUsing -s 2 may be faster (also see FAQ)\n\n");

	// calculate objective value

	double v = 0;
	int nSV = 0;
	for(i=0; i<w_size; i++)
		v += w[i]*w[i];
	for(i=0; i<l; i++)
	{
		v += alpha[i]*(alpha[i]*diag[GETI(i)] - 2);
		if(alpha[i] > 0)
			++nSV;
	}
	info("Objective value = %lf\n",v/2);
	info("nSV = %d\n",nSV);

	delete [] QD;
	delete [] alpha;
	delete [] y;
	delete [] index;

	return w;
}

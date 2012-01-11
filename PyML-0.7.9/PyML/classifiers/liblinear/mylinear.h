
// PyML includes:
# include "../../containers/ext/SparseDataSet.h"
# include "../../containers/ext/SparseFeatureVector.h"
# include <vector>

enum { L2R_LR, L2R_L2LOSS_SVC_DUAL, L2R_L2LOSS_SVC, L2R_L1LOSS_SVC_DUAL, MCSVM_CS, L1R_L2LOSS_SVC, L1R_LR, L2R_LR_DUAL }; /* solver_type */

std::vector<double> solve_l2r_l1l2_svc(SparseDataSet *data, double eps, 
				       double Cp, double Cn, int solver_type);



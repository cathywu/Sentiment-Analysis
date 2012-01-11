import numpy
from PyML.classifiers.composite import CompositeClassifier
from PyML.classifiers.baseClassifiers import Classifier
from PyML.evaluators import assess
from PyML.datagen import sample

class Platt (CompositeClassifier) :
    """
    Converts a real valued classifier into a conditional probability estimator.
    This is achieved by fitting a sigmoid with parameters A and B to the
    values of the decision function:
    f(x) -->  1/(1+exp(A*f(x)+B)

    code is a based on Platt's pseudocode from:

    John C. Platt.  Probabilistic Outputs for Support Vector
    Machines and Comparisons to Regularized Likelihood Methods. in:
    Advances in Large Margin Classifiers
    A. J. Smola, B. Schoelkopf, D. Schuurmans, eds. MIT Press (1999).

    :Keywords:
      - `mode` - values: 'holdOut' (default), 'cv'.
        The Platt object fits a sigmoid to the values of the classifier decision 
        function.  The values of the decision function are computed in one of two
        ways:  on a hold-out set (the 'holdOut' mode), or by cross-validation
        (the 'cv' mode).
      - `fittingFraction` - which fraction of the training data to use for fitting
        the sigmoid (the rest is used for the classifier training).  default: 0.2
      - `numFolds` - the number of cross-validation folds to use when in 'cv' mode.

    """
    
    attributes = {'mode' : 'holdOut',
                  'numFolds' : 3,
                  'fittingFraction' : 0.2}

    def train(self, data, **args) :

        Classifier.train(self, data, **args)
        if self.labels.numClasses != 2 :
            raise ValueError, 'number of classes is not 2'

        if self.mode == 'cv' :
            self.classifier.train(data, **args)

        numTries = 0
        maxNumTries = 5
        success = False
        while not success and numTries < maxNumTries :
            numTries += 1
            if self.mode == 'cv' :
                fittingData = data
                r = self.classifier.stratifiedCV(data, self.numFolds)
            elif self.mode == 'holdOut' :
                fittingData, trainingData = sample.splitDataset(data, self.fittingFraction)
                self.classifier.train(trainingData, **args)
                r = self.classifier.test(fittingData)
            else :
                raise ValueError, 'unknown mode for Platt'
	    self.labels = self.classifier.labels

        prior1 = fittingData.labels.classSize[1]
        prior0 = fittingData.labels.classSize[0]
        out = numpy.array(r.Y, numpy.float_)
        try :
            self.fit_A_B(prior1, prior0, out, r.decisionFunc, r.givenY)		
            success = True
        except :
            pass

        if not success :
            print 'platt not successful'
            self.A = None
            self.B = None
            results = self.classifier.test(data)
            maxPos = 1e-3
            minNeg = -1e-3
            for f in results.decisionFunc :
                if f > 0 :
                    if f > maxPos :
                        maxPos = f
                elif f < 0 :
                    if f < minNeg :
                        minNeg = f
            self.maxPos = maxPos
            self.minNeg = abs(minNeg)
        
        self.log.trainingTime = self.getTrainingTime()

    def fit_A_B(self, prior1, prior0, out, deci, Y) :
        
        A = 0.0
        B = math.log((prior0 + 1.0) / (prior1 + 1.0))
        hiTarget = (prior1 + 1.0) / (prior1 + 2.0)
        loTarget = 1.0 / (prior0 + 2.0)
        l = 1e-3
        olderr = 1e15

        pp = numpy.ones(len(data), numpy.float_) * \
             (prior1 + 1.0) / (len(data) + 2.0)

        count = 0
        t = numpy.zeros(len(data), numpy.float_)
        for i in range(len(data)) :
            if Y[i] == 1 :
                t[i] = hiTarget
            else :
                t[i] = loTarget

        for it in range(1,101) :
            d1 = pp - t
            d2 = pp * (1 - pp)
            a = numpy.sum(out * out * d2)
            b = numpy.sum(d2)
            c = numpy.sum(out * d1)
            d = numpy.sum(out * d1)
            e = numpy.sum(d1)
            if abs(d) < 1e-9 and abs(e) < 1e-9 :
                break
            oldA = A
            oldB = B
            err = 0.0
            while 1 :
                det = (a + l) * (b + l) - c * c
                if det == 0 :
                    l *= 10
                    continue
                A = oldA + ((b + l) * d - c * e) / det
                B = oldB + ((a + l) * e - c * d) / det

                pp = 1.0 / (1 + numpy.exp(out * A + B))
                pp2 = 1.0 / (1 + numpy.exp(-out * A - B))
                err = - numpy.sum(t * numpy.log(pp) +
                                    (1-t) * numpy.log(pp2))
                if err < olderr * (1 + 1e-7) :
                    l *= 0.1
                    break

                l *= 10
                if l > 1e6 :
                    raise ValueError, 'lambda too big'
            diff = err - olderr
            scale = 0.5 * (err + olderr + 1.0)
            if diff > -1e-3*scale and diff < 1e-7 * scale :
                count += 1
            else :
                count = 0
            olderr = err
            if count == 3 :
                break
        
        self.A = A
        self.B = B
        self.log.trainingTime = self.getTrainingTime()

    def decisionFunc(self, data, i) :

        f = self.classifier.decisionFunc(data, i)
        if self.A is not None :
            return 1.0 / (1 + math.exp(self.A * f + self.B))
        else :
            if f > 0 :
                return f / self.maxPos
            else :
                return f / self.minNeg
        
    def classify(self, data, i) :

        prob = self.decisionFunc(data ,i)
        if prob > 0.5 :
            return (1,prob)
        else:
            return (0,prob)

    test = assess.test
    
    def save(self, fileName) :

        if type(fileName) == type('') :
            outfile = open(fileName, 'w')
        else :
            outfile = fileName

        outfile.write('#A=' + str(self.A) + '\n')
        outfile.write('#B=' + str(self.B) + '\n')

        self.classifier.save(outfile)

    def load(self, fileName) :

        A = None
        B = None
        infile = open(fileName)
        for line in infile :
            if line.find('A=') > 0 :
                self.A = float(line[3:])
            if line.find('B=') > 0 :
                self.B = float(line[3:])
                break
        infile.close()
        self.classifier = svm.loadSVM(fileName)
        self.labels = self.classifier.labels

class Platt2 (Platt) :
    '''
    Converts a real valued classifier into a conditional probability estimator.
    This is achieved by fitting a sigmoid with parameters A and B to the
    values of the decision function:
    f(x) -->  1/(1+exp(A*f(x)+B)
    
    The fitting procedure is a Levenberg-Marquardt
    optimization derived by Tobias Mann using
    Mathematica, to optimize the objective function
    in:
    
    John C. Platt.  Probabilistic Outputs for Support Vector
    Machines and Comparisons to Regularized Likelihood Methods. in:
    Advances in Large Margin Classifiers
    A. J. Smola, B. Schoelkopf, D. Schuurmans, eds. MIT Press (1999).
    '''
    
    def fit_A_B(self, prior1, prior0, out, deci, Y) :
    
        hiTarget = (prior1 + 1.0) / (prior1 + 2.0)
        loTarget = 1.0 / (prior0 + 2.0)
        t = numpy.zeros(len(Y), numpy.float_)
        for i in range(len(Y)) :
            if Y[i] == 1 :
                t[i] = hiTarget
            else :
                t[i] = loTarget
                
        maxiter = 100
        minstep = 1e-10
        sigma = 1e-3
        A = 0.0
        B = math.log((prior0 + 1.0) / (prior1 + 1.0))
        A_init = A
        B_init = B
        ll = self.log_likelihood(t,deci,A,B)
        lm_lambda = 1e-4
        for it in range(maxiter) :
            H = self.hessian(t,deci,A,B)
            grad = self.gradient(t,deci,A,B)
            H_for_inversion = H
            H_for_inversion[0][0] = H_for_inversion[0][0]+lm_lambda
            H_for_inversion[1][1] = H_for_inversion[1][1]+lm_lambda
            cond = self.condition_number(H_for_inversion)
            if cond is None or cond > 1e5:
                A = A_init
                B = B_init
                break
            
            inverse_H = self.two_by_two_inverse( H_for_inversion )
            update_vec = self.get_proposed_update_vec(inverse_H, grad)
            proposed_ll = self.log_likelihood(t,deci,A-update_vec[0],
                                         B-update_vec[1])
            if proposed_ll < ll:
                A = A-update_vec[0]
                B = B-update_vec[1]
                lm_lambda = lm_lambda/10
                delta = ll-proposed_ll
                ll = proposed_ll
                if delta < 1e-4:
                    break
            else:
                lm_lambda = lm_lambda * 10
                
        self.A = A
        self.B = B

    def get_proposed_update_vec(self,m, v):
        update_vec = [0,0]
        update_vec[0] = m[0][0]*v[0]+m[0][1]*v[1]
        update_vec[1] = m[1][0]*v[0]+m[1][1]*v[1]
        return update_vec
    
    def condition_number(self,M):
        # assumes 2x2 matrices!
        M_inverse = self.two_by_two_inverse(M)

        if M_inverse is None:
            condition_number = None
        else:
            M_norm = math.sqrt(M[0][0]**2+
                               M[0][1]**2+
                               M[1][0]**2+
                               M[1][1]**2)
            M_inverse_norm = math.sqrt(M_inverse[0][0]**2+
                                       M_inverse[0][1]**2+
                                       M_inverse[1][0]**2+
                                       M_inverse[1][1]**2)
            condition_number = M_norm*M_inverse_norm

        return condition_number
                                     
                                     
        
    def log_likelihood(self,t,f,A,B):
        # computes Platt's log likelihood
        # function.  t is the target vector,
        # f is the decision function vector, A and B
        # are the sigmoid parameters

        ll = 0
        small = 1e-15
        
        for i in range(len(t)):
            exp_term = math.exp(A*f[i]+B)
            p_i = 1/(1+exp_term)

            # don't take the log of zero!
            if p_i < small:
                p_i = small

            # also trouble if 1-p_i = 0...
            if abs(p_i-1) < small:
                p_i = 1-small

            ll = ll + t[i]*math.log(p_i) + \
                 (1-t[i])*math.log(1-p_i)

        return -ll

    def two_by_two_inverse(self,M):
        # for the 2x2 matrix M,
        # with elements:
        # [a,b
        #  c,d], the inverse
        # is
        # 1/(ad-bc) * [d -b; -c a]

        a = M[0][0]
        b = M[0][1]
        c = M[1][0]
        d = M[1][1]
        det = a*d-b*c
        I = [[0,0],[0,0]]
        if det == 0:
            I = None
        else:
            I[0][0] = d/det
            I[0][1] = -b/det
            I[1][0] = -c/det
            I[1][1] = a/det
        return I
    
    def gradient(self,t,f,A,B):
        gradient = [0,0]
        gradient[0] = self.dF_dA(t,f,A,B)
        gradient[1] = self.dF_dB(t,f,A,B)

        return gradient
    
    def hessian(self,t,f,A,B):
        d2f_dA2 = self.dF_dAA(t,f,A,B)
        d2f_dB2 = self.dF_dBB(t,f,A,B)
        d2f_dAB = self.dF_dAB(t,f,A,B)
        
        hessian = [[0,0],[0,0]]
        hessian[0][0] = d2f_dA2
        hessian[0][1] = d2f_dAB
        hessian[1][0] = d2f_dAB
        hessian[1][1] = d2f_dB2

        return hessian
    
    def dF_dA(self,t,f,A,B):
        # computes the partial derivative
        # of F (the log likelihood) w.r.t.
        # A
        small = 1e-15
        partial = 0
        for i in range(len(t)):
            invprob = 1+math.exp(B+A*f[i])
            prob = 1/invprob
            if abs(prob-1) < small:
                prob = 1-small
                
            partial = partial + \
                      (math.exp(B+A*f[i])*f[i]*(1-t[i]))/ \
                      (invprob**2 * (1-prob)) - \
                      math.exp(B+A*f[i])*prob*f[i]*t[i]
            
        return -partial
    
    def dF_dB(self,t,f,A,B):
        # computes the partial derivative
        # of F (the log likelihood) w.r.t.
        # B
        small = 1e-15
        partial = 0
        for i in range(len(t)):
            invprob = 1+math.exp(B+A*f[i])
            prob = 1/invprob
            if abs(prob-1) < small:
                prob = 1-small
                
            partial = partial + \
                      (math.exp(B+A*f[i])*(1-t[i]))/ \
                      (invprob**2 * (1-prob)) - \
                      math.exp(B+A*f[i])*prob*t[i]
            
        return -partial

    def dF_dAA(self,t,f,A,B):
        # computes the second partial
        # derivative of F w.r.t. A
        small = 1e-15
        partial = 0
        for i in range(len(t)):
            invprob = 1+math.exp(B+A*f[i])
            prob = 1/invprob
            if abs(prob-1) < small:
                prob = 1-small
            partial = partial + \
                      -((math.exp(2*B + 2*A*f[i])*f[i]**2*(1 - t[i]))/(invprob**4*(1 - prob)**2)) - \
                      (2*math.exp(2*B + 2*A*f[i])*f[i]**2*(1 - t[i]))/(invprob**3*(1 - prob)) + \
                      (math.exp(B + A*f[i])*f[i]**2*(1 - t[i]))/(invprob**2*(1 - prob)) + \
                      (math.exp(2*B + 2*A*f[i])*f[i]**2*t[i])/invprob**2 - \
                      math.exp(B + A*f[i])*prob*f[i]**2*t[i]

        return -partial

    def dF_dBB(self,t,f,A,B):
        # computes the second partial
        # derivative of F w.r.t. A
        small = 1e-15
        partial = 0
        for i in range(len(t)):
            invprob = 1+math.exp(B+A*f[i])
            prob = 1/invprob
            if abs(prob-1) < small:
                prob = 1-small
            partial = partial + \
                      -((math.exp(2*B + 2*A*f[i])*(1 - t[i]))/(invprob**4*(1 - prob)**2)) - \
                      (2*math.exp(2*B + 2*A*f[i])*(1 - t[i]))/(invprob**3*(1 - prob)) + \
                      (math.exp(B + A*f[i])*(1 - t[i]))/(invprob**2*(1 - prob)) + \
                      (math.exp(2*B + 2*A*f[i])*t[i])/invprob**2 - math.exp(B + A*f[i])*prob*t[i]

        return -partial

    def dF_dAB(self,t,f,A,B):
        # computes the second partial
        # derivative of F w.r.t. A and B
        small = 1e-15
        partial = 0
        for i in range(len(t)):
            invprob = 1+math.exp(B+A*f[i])
            prob = 1/invprob
            if abs(prob-1) < small:
                prob = 1-small
            partial = partial + \
                      -((math.exp(2*B + 2*A*f[i])*f[i]*(1 - t[i]))/(invprob**4*(1 - prob)**2)) - \
                      (2*math.exp(2*B + 2*A*f[i])*f[i]*(1 - t[i]))/(invprob**3*(1 - prob)) + \
                      (math.exp(B + A*f[i])*f[i]*(1 - t[i]))/(invprob**2*(1 - prob)) + \
                      (math.exp(2*B + 2*A*f[i])*f[i]*t[i])/invprob**2 - math.exp(B + A*f[i])*prob*f[i]*t[i]

        return -partial

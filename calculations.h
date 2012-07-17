#ifndef M_CALCULATIONS
#define M_CALCULATIONS

#include <complex>
#include <iostream>

#include <Dense>
#include <Householder>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::HouseholderQR;

#define THETA 2.0
#define ALPHA 0.5
int num_clusters = 0;

double gammaln(double x) {
    double tmp, ser;
    static double cof[6] = {76.18009173, -86.50532033, 24.01409822,
                            -1.231739516, 0.120858003e-2, -0.536382e-5};
    int j;

    x -= 1.0;
    tmp = x + 5.5;
    tmp -= (x + 0.5)*log(tmp);
    ser = 1.0;
    for (j = 0; j <= 5; j++) {
        x += 1.0;
        ser += cof[j]/x;
    }
    return -tmp + log(2.50662827465*ser);
}

class ClusterStats {
    int d, n;
    double v0, k0;
    VectorXd u0;
    MatrixXd T0;
    VectorXd sum;
    MatrixXd sum_squared;

    ClusterStats(int d, double v0, double k0, VectorXd u0, MatrixXd T0) :
        d(d),
        n(0),
        v0(v0),
        k0(k0),
        u0(u0),
        T0(T0),
        sum(d),
        sum_squared(d,d) {}

    void add(VectorXd x) {
        if(n==0) num_clusters++;
        n++;
        sum += x;
        sum_squared += x*x.transpose();
    }

    void remove(VectorXd x) {
        n--;
        if(n==0) num_clusters--;
        sum -= x;
        sum_squared -= x*x.transpose();
    }
    
    void clear(){
      if(n!=0) num_clusters--;
      n = 0;
      sum = VectorXd(d);
      sum_squared = MatrixXd(d,d);
    }

    // O(d^2) to compute the logpdf
    double logpdf_em(VectorXd x) {
        VectorXd mu = sum/n;
        MatrixXd sigma = sum_squared/n - mu*mu.transpose();
        HouseholderQR<MatrixXd> qr(sigma);
        return -0.5*d*log(2*M_PI) - 0.5*qr.logAbsDeterminant() \
               - 0.5*(x - mu).transpose()*sigma.inverse()*(x - mu) + log(n + THETA);
    }

    // O(d^2) to compute all the parameters, O(d^3) to compute logtpdf
    // Equations:
    // x ~ t_{v_n-d+1}(u_n, T_n(k_n+1)/k_n(v_n-d+1))
    // u_n = (k*u_0 + n*xbar)/(k+n)
    // v_n = v + n
    // k_n = k + n
    // T_n = T + S_n + k*n/(k+n)*(u0-xbar)*(u0-xbar)^T
    // S_n = sum (x_i-xbar)*(x_i-xbar)^T
    // sufficient statistics for all of this:
    // sum x_i
    // sum x_i*x_i^T
    // n
    double logpdf_mcmc(VectorXd x) {
        if (n > 0) {
            VectorXd xbar = sum/n;
            VectorXd un = (k0*u0 + n*xbar)/(k0 + n);
            double vn = v0 + n;
            double kn = k0 + n;
            MatrixXd Sn = sum_squared - sum*sum.transpose()/n;
            MatrixXd Tn = T0 + Sn + k0*n/(k0 + n)*(u0 - xbar)*(u0 - xbar).transpose();
            return logtpdf(x, d, vn - d + 1, un, Tn*(kn + 1)/(kn*(vn - d + 1))) + log(n-ALPHA);
        } else {
            return logtpdf(x, d, v0 - d + 1, u0, T0*(k0 + 1)/(k0*(v0 - d + 1))) + log(THETA+num_clusters*ALPHA);
        }
    }

    // t(x) = Gamma(v/2+d/2)/Gamma(v/2) * det(Sigma)^-0.5 \
    // / [(pi*v)^(d/2)] * [1+(1/v)*(x-u)^T*Sigma^-1*(x-u)]^-((v+d)/2)
    double logtpdf(VectorXd x, int d, double v, VectorXd mu, MatrixXd sigma){
        HouseholderQR<MatrixXd> qr(sigma);
        return gammaln((v + d)/2.0) - gammaln(v/2.0) \
               - 0.5*qr.logAbsDeterminant() - 0.5*d*log(M_PI*v) \
               - 0.5*(v + d)*log(1 + (1.0/v)*(x - mu).transpose()*sigma.inverse()*(x - mu));
    }
};

#endif

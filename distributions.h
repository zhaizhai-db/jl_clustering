#ifndef M_DISTRIBUTIONS
#define M_DISTRIBUTIONS
#include <ctime>
#include <cstdlib>
#include <cmath>

#include <Dense>

using Eigen::VectorXd;

bool RAND_INITIALIZED = false; // maybe we shouldn't have this global...

void init_random() {
  if (!RAND_INITIALIZED) {
    RAND_INITIALIZED = true;
    srandom(59403854);
  }
}

// returns uniformly randomom real in [0, max]
double random_double(double max_val=1.0) {
  init_random();
  return max_val * rand() / (double) RAND_MAX;
}

// returns uniformly randomom integer in the interval [0, x)
int random_int(int x) {
  init_random();
  return rand() % x;
}

double gaussian() {
  double u1 = 1 - random_double(1.0), u2 = 1 - random_double(1.0);
  return sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
}

double gaussian(double mean, double stddev) {
  double g = gaussian();
  g *= stddev;
  g += mean;
  return g;
}

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

// returns the logpdf of a standard d-dimensional Gaussian at x for
// |x|^2 = norm_squared
double gaussian_logpdf(double norm_squared, int dim) {
    return -0.5 * (dim * log (2 * M_PI) + norm_squared);
}

#endif

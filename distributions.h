#ifndef M_DISTRIBUTIONS
#define M_DISTRIBUTIONS
#include <ctime>
#include <cstdlib>
#include <cmath>
#include <vector>

#include <Dense>

using Eigen::VectorXd;
using std::vector;

bool RAND_INITIALIZED = false; // maybe we shouldn't have this global...
double ANNEAL_TEMPERATURE = 1.0;
void init_anneal(double _temperature){
    ANNEAL_TEMPERATURE = _temperature;
}
void step_anneal(double multiplier){
    assert(ANNEAL_TEMPERATURE > 0);
    ANNEAL_TEMPERATURE *= multiplier;
    assert(ANNEAL_TEMPERATURE > 0);
}
void stop_anneal(){
    init_anneal(1.0);
}

void init_random() {
  if (!RAND_INITIALIZED) {
    RAND_INITIALIZED = true;
    //srand(59403854);
    srand(time(NULL));
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

// numerically robust sampling from logprobs
// e.g. if logprobs = [0, -1, -2], then
// sum = 1 + e^-1 + e^-2, and
// returns 0 with probability 1/sum, etc.
int sample(vector<double> logprobs){
    for(int i=0;i<(int)logprobs.size();i++)
        logprobs[i] /= ANNEAL_TEMPERATURE;
    double largest = logprobs[0];
    for(int i=1;i<(int)logprobs.size();i++){
        largest = std::max(largest,logprobs[i]);
    }
    double sum = 0.0;
    for(int i=0;i<(int)logprobs.size();i++){
        //cout << "logprobs[" << i << "]=" << logprobs[i] << endl;
        logprobs[i] -= largest;
        sum += exp(logprobs[i]);
    }
    double u = random_double(sum), partial_sum = 0.0;
    //cout << "u=" << u << " sum=" << sum << endl;
    for(int i=0;i<(int)logprobs.size();i++){
        partial_sum += exp(logprobs[i]);
        //cout << "partial sum=" << partial_sum << endl;
        if(u < partial_sum){
            return i;
        }
    }
    assert(false);
}

#endif

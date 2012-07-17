#ifndef M_DISTRIBUTIONS
#define M_DISTRIBUTIONS
#include <ctime>
#include <cstdlib>
#include <cmath>

bool RAND_INITIALIZED = false; // maybe we shouldn't have this global...

void init_random() {
  if (!RAND_INITIALIZED) {
    RAND_INITIALIZED = true;
    srandom(time(NULL));
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

#endif

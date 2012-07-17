#ifndef M_DISTRIBUTIONS
#define M_DISTRIBUTIONS
#include <cstdlib>
#include <cmath>

// returns uniformly random real in [0, max]
double rand(double max_val) {
  return rand() / (double) RAND_MAX;
}

// returns uniformly random integer in the interval [0, x)
int randint(int x) {
  return rand() % x;
}

double gaussian() {
  double u1 = 1 - rand(1.0), u2 = 1 - rand(1.0);
  return sqrt(-2 * log(u1)) * cos(2 * M_PI * u2);
}

double gaussian(double mean, double stddev) {
  double g = gaussian();
  g *= stddev;
  g += mean;
  return g;
}

#endif

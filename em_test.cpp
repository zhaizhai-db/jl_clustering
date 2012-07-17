#include <iostream>
#include "em.h"

#include <Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

void test1(){
  MatrixXd data(5,2);
  data << 1.0,0.0,
          1.0,0.1,
          1.0,0.2,
          5.0,2.0,
          5.0,2.1;
  em(data,2,5);
}

int main(){
  test1();
}

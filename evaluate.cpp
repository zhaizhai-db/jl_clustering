#include <stdio.h>
#include <vector>
#include <assert.h>
#include <ctime>

#include "em.h"

#include <Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

//Run as: ./evaluate filename algorithm k0 v0 lambda
int main(int argc, char **argv) {
    char prefix[1000];
    sprintf(prefix,"data/");
    char filename[1000];
    assert(argc >= 6);
    sprintf(filename,"%s%s.train",prefix,argv[1]);
    int ALGORITHM;
    double K0, V0, LAMBDA;
    sscanf(argv[2],"%d",&ALGORITHM);
    sscanf(argv[3],"%lf",&K0);
    sscanf(argv[4],"%lf",&V0);
    sscanf(argv[5],"%lf",&LAMBDA);
    initialize_em(ALGORITHM, K0, V0, LAMBDA);

    printf("Reading training data...\n");
    FILE *ftrain = fopen(filename,"r");
    int N, D, K;
    fscanf(ftrain,"%d%d%d",&N,&D,&K);
    MatrixXd data_train = MatrixXd::Zero(N, D);
    vector<int> labels_train(N, -1);
    double temp_f; int temp_i;
    for(int n=0;n<N;n++){
        for(int d=0;d<D;d++){
            fscanf(ftrain,"%lf",&temp_f);
            data_train(n,d) = temp_f;
        }
        fscanf(ftrain,"%d",&temp_i);
        labels_train[n] = temp_i;
    }
    printf("Done reading training data, starting training...\n");
    
    long long int time1 = clock();
    vector<int> assignments;
    vector<Cluster*> clusters;
    if (em(data_train, K, labels_train, &assignments, &clusters, -1, false, 1) != 0) {
        printf("Error: em returned a non-zero status code.\n");
        return 1;
    }
    long long int time2 = clock();
    printf("Done training: took %.4f seconds.\n", (time2-time1)/(float)CLOCKS_PER_SEC);
    fclose(ftrain);

    printf("Reading holdout data...\n");
    sprintf(filename,"%s%s.holdout",prefix,argv[1]);
    FILE *fholdout = fopen(filename,"r");
    int N2, D2, K2;
    fscanf(fholdout,"%d%d%d",&N2,&D2,&K2);
    assert(D2==D);
    assert(K2==K);
    MatrixXd data_holdout = MatrixXd::Zero(N2,D2);
    vector<int> labels_holdout = vector<int>(N2,-1);
    for(int n=0;n<N2;n++){
        for(int d=0;d<D2;d++){
            fscanf(fholdout,"%lf",&temp_f);
            data_holdout(n,d)=temp_f;
        }
        /*printf("Log probabilities:");
        for(int k=0;k<K;k++)
            printf(" %.2f", clusters[k]->log_posterior(data_holdout.row(n)));
        printf("\n");*/
        fscanf(fholdout,"%d",&temp_i);
        labels_holdout[n]=temp_i;
    }
    printf("Done reading holdout data, starting predictions...\n");
    long long int time3 = clock();
    vector<int> labels_holdout_predicted;
    if(ALGORITHM==0){
        if (reassign_naive(data_holdout, clusters, vector<int>(N2,-1), 1, &labels_holdout_predicted) != 0) {
            printf("Error: reassign_naieve returned a non-zero status code.\n");
        }
    } else if(ALGORITHM==1){
        if (reassign_jl(data_holdout, clusters, vector<int>(N2,-1), 1, &labels_holdout_predicted) != 0) {
            printf("Error: reassign_jl returned a non-zero status code.\n");
        }
    } else {
        printf("Error: invalid algorithm choice %d.\n", ALGORITHM);
    }
    long long int time4 = clock();
    printf("Done assigning predictions: took %.4f seconds.\n", (time4-time3)/(float)CLOCKS_PER_SEC);

    int num_correct = 0;
    for(int n=0;n<N2;n++){
        if(labels_holdout_predicted[n] == labels_holdout[n]){
            num_correct++;
        }
    }
    printf("Got %d/%d correct.\n",num_correct,N2);
    printf("Training took %.4f seconds, predictions took %.4f seconds.\n", (time2-time1)/(float)CLOCKS_PER_SEC, (time4-time3)/(float)CLOCKS_PER_SEC);

    return 0;
}

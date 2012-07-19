GIT_HOME=`git rev-parse --show-toplevel`
DATA_HOME=${GIT_HOME}/data
DATASET_NAME="32x32"

for s in test holdout pretraining; do
    cp ${DATASET_NAME}_${s}.txt ${DATA_HOME}/${s}.txt
done
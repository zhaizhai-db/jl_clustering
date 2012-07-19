GIT_HOME=`git rev-parse --show-toplevel`
DATA_HOME=${GIT_HOME}/data
DATASET_NAME="32x32"

for s in test holdout pretrain; do
    cp ${DATASET_NAME}_${s}.txt ${DATA_HOME}/${DATASET_NAME}.${s}
done

cat ${DATA_HOME}/${DATASET_NAME}.pretrain | python ${DATA_HOME}/util/remove_training_labels.py > ${DATA_HOME}/${DATASET_NAME}.train
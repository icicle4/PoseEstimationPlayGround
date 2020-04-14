#!/usr/bin/env bash

# check the enviroment info
PYTHON="python"

WORK_DIR=$(cd $(dirname $0)/../../../;pwd)
export PYTHONPATH=${WORK_DIR}:${PYTHONPATH}
cd ${WORK_DIR}

MODEL_NAME="hrpose"
CHECKPOINTS_NAME="hrpose_res18_coco_pose"$2
CONFIG_FILE='configs/hr_pose_res50_coco_pose.conf'

LOG_DIR="./log/pose/coco/"
LOG_FILE="${LOG_DIR}${CHECKPOINTS_NAME}.log"

if [[ ! -d ${LOG_DIR} ]]; then
    echo ${LOG_DIR}" not exists!!!"
    mkdir -p ${LOG_DIR}
fi

if [[ "$1"x == "train"x ]]; then
  ${PYTHON} -u main.py --config_file ${CONFIG_FILE} --phase train --gpu 0 \
                       --model_name ${MODEL_NAME} \
                       --checkpoints_name ${CHECKPOINTS_NAME} 2>&1 | tee ${LOG_FILE}

elif [[ "$1"x == "val"x ]]; then
  ${PYTHON} -u main.py --config_file ${CONFIG_FILE} --phase test --model_name ${MODEL_NAME} \
                       --phase test --gpu 0 2>&1 | tee -a ${LOG_FILE}

else
  echo "$1"x" is invalid..."
fi

#!/bin/bash
# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
pip install ordered_set
if [ $# -lt 1 ]
then
    echo "Usage: bash scripts/EAL-GAN_eval_1p.sh [DATASE_FOLDER]"
exit 1
fi

get_real_path(){
  if [ "${1:0:1}" == "/" ]; then
    echo "$1"
  else
    echo "$(realpath -m $PWD/$1)"
  fi
}

DATASE_FOLDER=$(get_real_path $1)

act_func="relu"
index=0
batch_list=(5 134 89 14 47 118 53 176 1 2 151 27 74 155 89 151)
for DATASET_NAME in "lympho" "glass" "ionosphere" "arrhythmia" "pima" "vowels" "letter" "cardio" "musk" "optdigits" "satimage-2" "satellite" "pendigits" "annthyroid" "mnist" "shuttle"
do
   if [ $DATASET_NAME = "shuttle" -o $DATASET_NAME = "annthyroid" -o $DATASET_NAME = "mnist" ]
   then
       act_func="tanh"   
   fi
   model_weights="weights_$DATASET_NAME"
   echo  "start eval for dataset $DATASET_NAME, index $index"
   python -u TrainAndEval.py \
        --dis_activation_func=$act_func\
        --resume="True"\
        --resume_batch=${batch_list[index]}\
        --device="CPU"\
        --data_path=$DATASE_FOLDER \
        --data_name=$DATASET_NAME > log/log_val_$DATASET_NAME.txt 2>&1 &\
   index=`expr $index + 1` 
done

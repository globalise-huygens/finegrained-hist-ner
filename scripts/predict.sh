#!/bin/bash

set -e

hdir=$(cd $(dirname ${BASH_SOURCE}[0]) && cd .. && pwd)
testdatapath="data/AB/AB.json"

predict() {
  cfg=$1
  data=$2
  model=$3
  ckpt=$4
  [[ ! -d "$hdir/data/$data/$model" ]] && mkdir "$hdir/data/$data/$model"
  srun python $hdir/src/finetune.py predict -c $cfg --ckpt_path=$ckpt --data.dataset=${testdatapath} --data.predict_key=validation --data.data_pkl="${data}-${model}_validation.pkl"
  mv data/$data/$model/seqeval_report*.txt "$hdir/data/$data/$model/seqeval_report_validation.txt"
  srun python $hdir/src/finetune.py predict -c $cfg --ckpt_path=$ckpt --data.dataset=${testdatapath} --data.predict_key=test --data.data_pkl="${data}-${model}_test.pkl"
  mv data/$data/$model/seqeval_report*.txt "$hdir/data/$data/$model/seqeval_report_test.txt"
}

cfg=$hdir/cfg/B_ckpt.yaml

# replace with appropriate checkpoint path
ckpt="${hdir}/mlruns/xxx/xxx/checkpoints/trainB-globertise-epoch=xxx.ckpt"
predict $cfg B globertise $ckpt

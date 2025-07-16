#!/bin/bash

set -e

hdir=$(cd $(dirname ${BASH_SOURCE}[0]) && cd .. && pwd)
cfg=$hdir/cfg/B.yaml
cfg_ckpt=$hdir/cfg/B_ckpt.yaml

seed=("23052024" "52024230" "24230520")

run_exps() {
  python $hdir/src/finetune.py fit -c $cfg_ckpt --seed_everything=${seed[0]} \
    --trainer.logger.run_name="seed=${seed[0]}"
  python $hdir/src/finetune.py fit -c $cfg --seed_everything=${seed[1]} \
    --trainer.logger.run_name="seed=${seed[1]}"
  python $hdir/src/finetune.py fit -c $cfg --seed_everything=${seed[2]} \
    --trainer.logger.run_name="seed=${seed[2]}"
}

run_exps

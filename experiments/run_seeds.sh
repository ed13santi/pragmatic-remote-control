#!/bin/bash
# lane runner: run_seeds.sh "KIND:SEED KIND:SEED ..."   (sign-flip task, 3M steps each)
for job in $1; do
  kind=${job%%:*}; seed=${job##*:}
  SEED=$seed python experiments/jumps.py inverted_pendulum_switching $kind 3000000 > ${LOGDIR:-/tmp}/seed_${kind}_${seed}.log 2>&1
done

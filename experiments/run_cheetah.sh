#!/bin/bash
# lane runner: run_cheetah.sh "KIND KIND ..."   (half-cheetah, 3M steps each)
for kind in $1; do
  python experiments/jumps.py half_cheetah $kind 3000000 > ${LOGDIR:-/tmp}/cheetah_${kind}.log 2>&1
done

#!/bin/bash
# Learned state-dependent threshold on both jump tasks. usage: experiments/run_threshold_learned.sh LOGDIR
LOG=${1:-.}
for task in inverted_pendulum_switching inverted_pendulum_kicks; do
  [ -f "figures/jumps_${task}_threshold_learned_0.0.json" ] || \
    python experiments/jumps.py $task threshold_learned > "$LOG/jumps_${task}_threshold_learned.log" 2>&1
done

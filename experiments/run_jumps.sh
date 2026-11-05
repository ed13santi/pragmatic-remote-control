#!/bin/bash
# Runs every jump experiment in two parallel streams, skipping finished ones.
# usage: experiments/run_jumps.sh LOGDIR
LOG=${1:-.}
run() {
  task=$1; kind=$2; value=$3
  v=$(python -c "print(float('${value:-0}'))")
  [ -f "figures/jumps_${task}_${kind}_${v}.json" ] && return
  python experiments/jumps.py $task $kind $value > "$LOG/jumps_${task}_${kind}_${v}.log" 2>&1
}
stream() {
  task=$1
  run $task always
  run $task joint
  run $task joint_rich
  for th in 0.1 0.2 0.4 0.8 1.6; do run $task threshold $th; done
}
stream inverted_pendulum_kicks &
stream inverted_pendulum_switching &
wait

#!/bin/bash
# Runs every jump experiment in two parallel streams. usage: experiments/run_jumps.sh LOGDIR
LOG=${1:-.}
run() { python experiments/jumps.py "$@" > "$LOG/jumps_$(echo "$@" | tr ' /' '__').log" 2>&1; }
stream() {
  for task in "$@"; do
    run $task always
    run $task joint
    run $task joint_rich
    for th in 0.1 0.2 0.4 0.8 1.6; do run $task threshold $th; done
  done
}
stream inverted_pendulum_kicks &
stream inverted_pendulum_switching &
wait

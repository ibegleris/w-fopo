#!/bin/bash
echo 'starting...'
rm -r output*
rm -r *__*
source activate intel
export MKL_NUM_THREADS=1
kernprof -l -v src/main_oscillator.py single 2 1 1
rm main_oscillator.py.lprof
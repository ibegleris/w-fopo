#!/bin/bash
echo 'starting...'
rm -r output*
rm -r *__*
source activate intel
export MKL_NUM_THREADS=1
kernprof -l -v mm_gnlse_2D.py 0 1
rm mm_gnlse_2D.py.lprof

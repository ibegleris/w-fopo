#!/bin/bash
echo 'starting...'
rm -r output*
rm -r *__*
export MKL_NUM_THREADS=1
kernprof -l -v mm_gnlse_2D.py 

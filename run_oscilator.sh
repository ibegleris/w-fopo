#!/bin/bash
source activate intel
echo 'starting...'
rm -r output*
rm -r *__*
export MKL_NUM_THREADS=$1
echo "running with" $1 "MKL core for" $2 "rounds" 
python mm_gnlse_2D.py $2 0

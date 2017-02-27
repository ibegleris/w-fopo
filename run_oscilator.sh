#!/bin/bash
echo 'starting...'
rm -r output*
rm -r *__*
export MKL_NUM_THREADS=1
python mm_gnlse_2D.py

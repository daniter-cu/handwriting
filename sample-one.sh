#!/bin/bash

THEANO_FLAGS=device=cuda0 /var/opt/wba/apps/anaconda2/bin/python sample_model.py -l tmp.txt  -f ../gold/f_sampling.pkl -s "$1" -b $2 -n 4

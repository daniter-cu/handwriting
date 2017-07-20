#!/bin/bash

THEANO_FLAGS=device=cuda0 /var/opt/wba/apps/anaconda2/bin/python sample_model.py -l "$1"  -f ../gold/f_sampling.pkl -b $2

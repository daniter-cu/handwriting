#!/bin/bash -e

export DATA_PATH=/home/dev01/share/var/opt/wba/share/data/
export TMP_PATH=intermediate/

THEANO_FLAGS=device=cuda0 /var/opt/wba/apps/anaconda2/bin/python main_cond.py
#/var/opt/wba/apps/anaconda2/bin/python main_cond.py

#/var/opt/wba/apps/anaconda2/bin/python sample_model.py -f $TMP_PATH/handwriting/EXP_ID/f_sampling.pkl -s 'Sous le pont Mirabeau coule la Seine.' -b 0.7 -n 2

#!/bin/bash -e

export TMP_PATH=intermediate/

/var/opt/wba/apps/anaconda2/bin/python sample_model.py -f $TMP_PATH/handwriting/$1/f_sampling.pkl -s 'Sous le pont Mirabeau coule la Seine.' -b 0.7 -n 2

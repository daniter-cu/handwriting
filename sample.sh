#!/bin/bash -e

while read p; do
  /var/opt/wba/apps/anaconda2/bin/python sample_model.py -f intermediate/handwriting/69454295/f_sampling.pkl -s "$p" -b 0.7 -n 4
  f="$(mktemp png.XXXXXX)"
  mv a_just_now.png $f
done <~/words.txt

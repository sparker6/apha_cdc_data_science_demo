#!/bin/sh

## data directory 
sim_dir='output'
mkdir $output 

python3 distilbert_args_classweight.py train.csv test.csv pred.csv $output


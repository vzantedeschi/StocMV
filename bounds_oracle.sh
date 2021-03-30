#!/bin/bash
for r in 1 2
do
    for d in moons
    do
	for b in mcallester
	do
	    for m in 4 8 16 32
            do
	        python3 toy_oracle.py training.risk=$r dataset.distr=$d model.pred=stumps-uniform bound.type=$b model.M=$m dataset.N_train=50
            done
	done
    done
done

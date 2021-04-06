#!/bin/bash
for r in 1 2
do
    for d in moons
    do
		for b in seeger mcallester
		do
		    for m in 4 8 16 32
	            do
		        python3 toy_oracle.py num_trials=10 training.lr=0.1 training.risk=$r dataset.distr=$d model.pred=stumps-uniform bound.type=$b model.M=$m dataset.N_train=1000
	            done
		done
    done
done

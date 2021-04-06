#!/bin/bash
for r in MC exact
do
    for d in moons
    do
        for m in 4 8 16 32
	do
	    for b in seeger mcallester
	    do
		    python3 toy.py num_trials=10 training.risk=$r dataset.distr=$d model.pred=stumps-uniform bound.type=$b model.M=$m dataset.N_train=1000 training.lr=0.1

	    done
        done
    done
done

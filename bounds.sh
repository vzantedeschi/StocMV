#!/bin/bash
for r in exact
do
    for d in moons
    do
        for m in 4 8 16 32
	do
	    for b in seeger mcallester
	    do
		    python3 toy.py training.risk=$r dataset.distr=$d model.pred=stumps-uniform bound.type=$b model.M=$m dataset.N_train=50
	    done
        done
    done
done

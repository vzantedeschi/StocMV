#!/bin/bash
for n in 1 10 100 1000
do
    for l in 0.001 0.01 0.1 1 10
    do
        for r in MC exact
        do
            for m in 4 8 16 32 64
            do
                python3.6 toy.py model.M=$m model.pred=stumps-uniform training.lr=$l training.risk=$r dataset.N_train=$n
            done

            python3.6 toy.py model.M=$m model.pred=stumps-optimal training.lr=$l training.risk=$r dataset.N_train=$n
        done
    done
done
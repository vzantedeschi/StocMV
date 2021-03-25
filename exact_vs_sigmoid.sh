#!/bin/bash
for l in 0.001 0.01 0.1 1 10
do
    for r in MC exact
    do
        python3.6 toy.py model.pred=stumps-uniform training.lr=$l training.risk=$r
        python3.6 toy.py model.pred=stumps-optimal training.lr=$l training.risk=$r
    done
done
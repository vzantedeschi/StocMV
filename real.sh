for r in exact MC Rnd FO SO
do
	for d in SHUTTLE MNIST FASHION SENSORLESS PROTEIN PENDIGITS
	do
		python3 real.py dataset=$d model.M=100 bound.type=seeger training.risk=$r model.pred=rf
	done
done

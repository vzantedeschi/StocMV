for b in seeger
do
	for d in SHUTTLE MNIST FASHION SENSORLESS PROTEIN PENDIGITS
	do
		for m in 100
		do

			for r in Rnd exp SO
			do
				python3 real.py dataset=$d model.M=$m bound.type=$b training.risk=$r model.pred=rf
			done
		done

	done
done

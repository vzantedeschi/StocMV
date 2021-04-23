for b in seeger
do
	for d in MNIST PENDIGITS FASHION PROTEIN SENSORLESS SHUTTLE
	do
		for m in 10 100
		do

			for r in exact MC FO SO
			do
				python3 real.py dataset=$d model.M=$m bound.type=$b training.risk=$r num_workers=8
			done
		done

	done
done

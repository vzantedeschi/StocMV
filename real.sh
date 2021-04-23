for b in seeger
do
	for d in MNIST PENDIGITS FASHION PROTEIN SENSORLESS SHUTTLE
	do
		for r in exact MC FO SO
		do
			python3.6 real.py dataset.name=$d model.M=10 bound.type=$b training.risk=$r num_workers=8
		done

	done
done

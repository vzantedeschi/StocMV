for r in exact MC Rnd FO SO
do
	# data-dependent prior
	for d in SHUTTLE MNIST FASHION SENSORLESS PROTEIN PENDIGITS
	do
		python3 real.py dataset=$d model.M=100 training.risk=$r model.pred=rf
	done

	# data-independent prior
	for d in HABER TTT SVMGUIDE CODRNA ADULT MUSH PHIS
	do
		python3 real.py dataset=$d model.M=10 training.risk=$r model.pred=stumps-uniform
	done
done

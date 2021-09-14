for r in FO SO Rnd MC exact
do
	# data-dependent prior
	for d in SENSORLESS PROTEIN PENDIGITS SHUTTLE MNIST FASHION 
	do
		python3 real.py dataset=$d model.M=100 training.risk=$r model.pred=rf
	done

	# data-independent prior
	for d in HABER TTT SVMGUIDE CODRNA ADULT MUSH PHIS
	do
		python3 real.py dataset=$d model.M=10 training.risk=$r model.pred=stumps-uniform
	done
done

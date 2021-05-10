for r in SO FO exp Rnd
do
	for d in MUSH TTT HABER SVMGUIDE PHIS CODRNA ADULT
	do
		python3 real.py dataset=$d model.uniform=True training.risk=$r model.pred=stumps-uniform model.M=10
	done

	for d in PENDIGITS SHUTTLE SENSORLESS PROTEIN FASHION MNIST
	do
		python3 real.py dataset=$d model.uniform=True training.risk=$r model.pred=rf model.M=100
	done
done

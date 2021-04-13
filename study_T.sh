for m in 64 128 256 512 1024
do
	for b in seeger
	do
		for t in 1 10 100
		do
			python3.6 toy.py num_trials=10 training.risk=MC training.lr=0.01 training.MC_draws=$t dataset.N_train=1000 bound.type=$b dataset.distr=moons training.seed=02042021 model.pred=stumps-uniform model.M=$m
		done

		python3.6 toy.py num_trials=10 training.risk=exact training.lr=0.01 dataset.N_train=1000 bound.type=$b dataset.distr=moons training.seed=02042021 model.pred=stumps-uniform model.M=$m
	done
done

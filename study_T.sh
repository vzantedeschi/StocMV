for m in 1 2 4 8 16 32
do
	for b in seeger
	do
		for t in 1 5 10 20 40 80 100
		do
			python3 toy.py num_trials=10 training.risk=MC training.lr=0.01 training.MC_draws=$t dataset.N_train=1000 bound.type=$b dataset.distr=moons training.seed=02042021 model.pred=stumps-uniform model.M=$m
		done

		python3 toy.py num_trials=10 training.risk=exact training.lr=0.01 dataset.N_train=1000 bound.type=$b dataset.distr=moons training.seed=02042021 model.pred=stumps-uniform model.M=$m
	done
done

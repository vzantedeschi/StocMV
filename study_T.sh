for n in 10 100 1000 10000
do
	for b in seeger mcallester
	do
		for t in 1 5 10 20 40 80 160
		do
			python3 toy.py num_trials=10 training.risk=MC training.lr=0.01 training.MC_draws=$t dataset.N_train=$n bound.type=$b dataset.distr=moons training.seed=02042021
		done

		python3 toy.py num_trials=10 training.risk=exact training.lr=0.01 dataset.N_train=$n bound.type=$b dataset.distr=moons training.seed=02042021
	done
done

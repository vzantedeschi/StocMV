for m in 1 2 3 4 5 6 7 8 9 10 138 12 14 655 16 19 22 25 281 29 159 33 39 44 429 51 184 568 59 323 68 79 212 91 868 104 1000 494 754 244 372 120
do
	for b in seeger
	do
		for t in 1 10 100
		do
			python3 toy.py num_trials=10 training.risk=MC training.lr=0.01 training.MC_draws=$t dataset.N_train=1000 bound.type=$b dataset.distr=moons training.seed=02042021 model.pred=stumps-uniform model.M=$m
		done

		python3 toy.py num_trials=10 training.risk=exact training.lr=0.01 dataset.N_train=1000 bound.type=$b dataset.distr=moons training.seed=02042021 model.pred=stumps-uniform model.M=$m
	done
done

for n in 10 50 100 200 300 400 500 600 700 800 900 1000 1100 1200 1300 1400 1500 1600 1700 1800 1900 2000
do 
	for b in seeger
	do
		for t in 100 500 1000
		do
			python3 toy.py num_trials=10 training.risk=MC training.lr=0.01 training.MC_draws=$t dataset.N_train=$n bound.type=$b dataset.distr=moons training.seed=02042021 model.pred=stumps-uniform model.M=4
		done

		python3 toy.py num_trials=10 training.risk=exact training.lr=0.01 dataset.N_train=$n bound.type=$b dataset.distr=moons training.seed=02042021 model.pred=stumps-uniform model.M=4
	done
done
for b in seeger mcallester
do
	for d in MUSH TTT SVMGUIDE HABER PHIS CODRNA ADULT HIGGS CLICK
	do
		for r in exact MC
		do
			python3.6 real.py dataset.name=$d model.M=10 bound.type=$b training.risk=$r num_workers=8
		done

		# for r in 1 2
		# do
		# 	python3 real_oracle.py dataset.name=$d model.M=10 bound.type=$b training.risk=$r num_workers=8
		# done
	done
done

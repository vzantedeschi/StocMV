# Learning Stochastic Majority Votes by Minimizing a PAC-Bayes Generalization Bound

### Dependencies

Running with python3.6.

Install PyTorch, following the [guidelines](https://pytorch.org/get-started/locally/).
Then install the requirements:

```bash
pip3 install -r requirements.txt
```

### Train on toy datasets
```bash
python3 toy.py
```

Default configuration is stored in 'config/toy.yaml'. You can edit directly the config file or change values from the command line, e.g. as follows: 
```bash
python3 toy.py dataset.N_train=1000 dataset.noise=0.1 model.M=16
```
See [Hydra](https://hydra.cc/docs/intro/) for a tutorial.

### Reproduce main results
To reproduce the main results of the paper on real benchmarks, run:
```bash
bash real.sh
```

You can also run a specific experiment, passing the chosen values for the hyper-parameters as follows:
```bash
python3 real.py dataset=SENSORLESS model.M=100 training.risk=MC model.pred=rf model.prior=2 model.tree_depth=5
```

### Minimal script
To run a simplified script that supports only the optimization of the proposed "exact" and "MC" bounds, run:
```bash
python3 minimal.py exact
python3 minimal.py MC
```

### Bibtex
If you find this work useful, please cite:

```
@article{zantedeschi2021learning,
  title={Learning Stochastic Majority Votes by Minimizing a PAC-Bayes Generalization Bound},
  author={Zantedeschi, Valentina and Viallard, Paul and Morvant, Emilie and Emonet, R{\'e}mi and Habrard, Amaury and Germain, Pascal and Guedj, Benjamin},
  journal={NeurIPS},
  year={2021}
}
```

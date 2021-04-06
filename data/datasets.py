import torch
from pathlib import Path

from data.real_datasets import *
from data.toy_datasets import *

REAL_DATASETS = {
    'A9A': fetch_A9A,
    'EPSILON': fetch_EPSILON,
    'PROTEIN': fetch_PROTEIN,
    'YEAR': fetch_YEAR,
    'HIGGS': fetch_HIGGS,
    'MICROSOFT': fetch_MICROSOFT,
    'YAHOO': fetch_YAHOO,
    'CLICK': fetch_CLICK,
    'GLASS': fetch_GLASS,
    'COVTYPE': fetch_COVTYPE,
    'DIGITS': fetch_DIGITS,
    'MUSH': fetch_MUSHROOMS,
    'TTT': fetch_TICTACTOE,
}

TOY_DATASETS = [
    'normals',
    'moons',
]

class Dataset:

    """
    Code adapted from https://github.com/Qwicen/node/blob/master/lib/data.py .

    """
    def __init__(self, dataset, data_path='./DATA', normalize=False, flatten=False, **kwargs):
        """
        Dataset is a dataclass that contains all training and evaluation data required for an experiment
        :param dataset: a pre-defined dataset name (see DATASETS) or a custom dataset
            Your dataset should be at (or will be downloaded into) {data_path}/{dataset}
        :param data_path: a shared data folder path where the dataset is stored (or will be downloaded into)
        :param normalize: standardize features by removing the mean and scaling to unit variance
        :param kwargs: depending on the dataset, you may select train size, test size or other params
        """

        if dataset in REAL_DATASETS:
            data_dict = REAL_DATASETS[dataset](Path(data_path) / dataset, **kwargs)

            self.X_valid = data_dict['X_valid']
            self.y_valid = data_dict['y_valid']
            
        elif dataset in TOY_DATASETS:
            data_dict = toy_dataset(name=dataset, **kwargs)

        else:
            raise NotImplementedError("Dataset not supported atm")

        self.X_train = data_dict['X_train']
        self.y_train = data_dict['y_train']
        self.X_test = data_dict['X_test']
        self.y_test = data_dict['y_test']

        if flatten:
            self.X_train, self.X_valid, self.X_test = self.X_train.reshape(len(self.X_train), -1), self.X_valid.reshape(len(self.X_valid), -1), self.X_test.reshape(len(self.X_test), -1)

        if normalize:

            print("Normalize dataset")
            axis = [0] + [i + 2 for i in range(self.X_train.ndim - 2)]
            self.mean = np.mean(self.X_train, axis=tuple(axis), dtype=np.float32)
            self.std = np.std(self.X_train, axis=tuple(axis), dtype=np.float32)

            # if constants, set std to 1
            self.std[self.std == 0.] = 1.

        self.data_path = data_path
        self.dataset = dataset

class TorchDataset(torch.utils.data.Dataset):

    def __init__(self, *data, **options):
        
        n_data = len(data)
        if n_data == 0:
            raise ValueError("At least one set required as input")

        self.data = data
        means = options.pop('means', None)
        stds = options.pop('stds', None)

        if options:
            raise TypeError("Invalid parameters passed: %s" % str(options))
         
        if means is not None:
            assert stds is not None, "must specify both <means> and <stds>"

            self.normalize = lambda data: [(d - m) / s for d, m, s in zip(data, means, stds)]

        else:
            self.normalize = lambda data: data

    def __len__(self):

        return len(self.data[0])

    def __getitem__(self, idx):

        data = self.normalize([s[idx] for s in self.data])
            
        return data
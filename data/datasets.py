import torch
from pathlib import Path

from data.real_datasets import *
from data.toy_datasets import *

BINARY_DATASETS = {
    'MUSH': fetch_MUSHROOMS,
    'TTT': fetch_TICTACTOE,
    'HABER': fetch_HABERMAN,
    'PHIS': fetch_PHISHING,
    'ADULT': fetch_ADULT,
    'CODRNA': fetch_CODRNA,
    'SVMGUIDE': fetch_SVMGUIDE1
}

MC_DATASETS = {
    'MNIST': fetch_MNIST,
    'PENDIGITS': fetch_PENDIGITS,
    'PROTEIN': fetch_PROTEIN,
    'SENSORLESS': fetch_SENSORLESS,
    'SHUTTLE': fetch_SHUTTLE,
    'FASHION': fetch_FASHION_MNIST,
    # 'USPS': fetch_USPS,
}

TOY_DATASETS = [
    'normals',
    'moons',
]

class Dataset:

    """
    Code adapted from https://github.com/Qwicen/node/blob/master/lib/data.py .

    """
    def __init__(self, dataset, data_path='./data', normalize=False, **kwargs):
        """
        Dataset is a dataclass that contains all training and evaluation data required for an experiment
        :param dataset: a pre-defined dataset name (see DATASETS) or a custom dataset
            Your dataset should be at (or will be downloaded into) {data_path}/{dataset}
        :param data_path: a shared data folder path where the dataset is stored (or will be downloaded into)
        :param normalize: standardize features by removing the mean and scaling to unit variance
        :param kwargs: depending on the dataset, you may select train size, test size or other params
        """

        path = Path(data_path)
        path.mkdir(parents=True, exist_ok=True)

        valid_size = kwargs.pop("valid_size", 0.2)
        if dataset in BINARY_DATASETS:

            self.binary = True
            data_dict = BINARY_DATASETS[dataset](path / dataset, valid_size=valid_size, **kwargs)

        elif dataset in MC_DATASETS:

            self.binary = False
            data_dict = MC_DATASETS[dataset](path / dataset, valid_size=valid_size, **kwargs)
            
        elif dataset in TOY_DATASETS:

            self.binary = True
            data_dict = toy_dataset(name=dataset, valid_size=valid_size, **kwargs)
            normalize = False
            valid_size = 0

        else:
            raise NotImplementedError("Dataset not supported")

        self.y_valid = data_dict.pop('y_valid', None)
        self.X_valid = data_dict.pop('X_valid', None)

        self.X_train = data_dict['X_train']
        self.y_train = data_dict['y_train']
        self.X_test = data_dict['X_test']
        self.y_test = data_dict['y_test']

        if normalize:

            print("Normalize dataset")
            axis = [0] + [i + 2 for i in range(self.X_train.ndim - 2)]
            self.mean = np.mean(self.X_train, axis=tuple(axis), dtype=np.float32)
            self.std = np.std(self.X_train, axis=tuple(axis), dtype=np.float32)

            # if constants, set std to 1
            self.std[self.std == 0.] = 1.

            self.X_train = (self.X_train - self.mean) / self.std
            self.X_test = (self.X_test - self.mean) / self.std

            if valid_size > 0:
                self.X_valid = (self.X_valid - self.mean) / self.std

        self.data_path = data_path
        self.dataset = dataset

class TorchDataset(torch.utils.data.Dataset):

    def __init__(self, X, y):
        
        self.num_data = len(X)

        self.X = X
        self.y = y[:, None]

    def __len__(self):

        return self.num_data

    def __getitem__(self, idx):
        # import pdb; pdb.set_trace()
        return self.X[idx], self.y[idx]

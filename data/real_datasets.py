import numpy as np
import gzip
import shutil
import tarfile
import os
import pandas as pd

import warnings

from pathlib import Path

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

from category_encoders import LeaveOneOutEncoder
from category_encoders.ordinal import OrdinalEncoder

from core.utils import download, read_idx_file

# BINARY CLASSIFICATION

def fetch_SVMGUIDE1(path, valid_size=0.2, seed=None):

    path = Path(path)
    train_path = path / 'svmguide1.data'
    test_path = path / 'svmguide1-test.data'

    if not train_path.exists() or not test_path.exists():
        path.mkdir(parents=True)

        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/svmguide1', train_path)
        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/svmguide1.t', test_path)

    X_train, y_train = read_idx_file(train_path, 4, " ")
    X_test, y_test = read_idx_file(train_path, 4, " ")

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size, random_state=seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_CODRNA(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    data_path = path / 'codrna.data'

    if not data_path.exists():
        path.mkdir(parents=True)

        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/cod-rna', data_path)

    X, Y = read_idx_file(data_path, 8)
    Y[Y == -1] = 0

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size / (1 - test_size), random_state=seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_PHISHING(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    data_path = path / 'phishing.data'

    if not data_path.exists():
        path.mkdir(parents=True)

        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/phishing', data_path)

    X, Y = read_idx_file(data_path, 68, " ")

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size / (1 - test_size), random_state=seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_ADULT(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    train_path = path / 'adult.data'
    test_path = path / 'adult.test'

    if not train_path.exists() or not test_path.exists():
        path.mkdir(parents=True)

        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a', train_path)
        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a1a.t', test_path)

    X_train, y_train = read_idx_file(train_path, 123)
    y_train[y_train == -1] = 0
    X_test, y_test = read_idx_file(test_path, 123)
    y_test[y_test == -1] = 0

    X, Y = np.vstack([X_train, X_test]), np.hstack([y_train, y_test])

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size / (1 - test_size), random_state=seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_HABERMAN(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    data_path = path / 'haberman.data'

    if not data_path.exists():
        path.mkdir(parents=True)

        download('https://archive.ics.uci.edu/ml/machine-learning-databases/haberman/haberman.data', data_path)

    data = np.genfromtxt(data_path, delimiter=',')

    X, Y = (data[:, 1:-1]).astype(np.float32), (data[:, -1] - 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size / (1 - test_size), random_state=seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_MUSHROOMS(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    data_path = path / 'agaricus-lepiota.data'

    if not data_path.exists():
        path.mkdir(parents=True, exist_ok=True)

        download('https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data', data_path)

    data = pd.read_csv(data_path, names=np.arange(23))
    encoder = OrdinalEncoder(return_df=False)
    data = encoder.fit_transform(data)
    
    X, Y = (data[:, 1:]).astype(np.float32), (data[:, 0] - 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size / (1 - test_size), random_state=seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_TICTACTOE(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    data_path = path / 'tic-tac-toe.data'

    if not data_path.exists():
        path.mkdir(parents=True, exist_ok=True)

        download('https://archive.ics.uci.edu/ml/machine-learning-databases/tic-tac-toe/tic-tac-toe.data', data_path)

    data = pd.read_csv(data_path, names=np.arange(10))
    encoder = OrdinalEncoder(return_df=False)
    data = encoder.fit_transform(data)
    
    X, Y = (data[:, :-1]).astype(np.float32), (data[:, -1] - 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size / (1 - test_size), random_state=seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

# MULTICLASS CLASSIFICATION
def fetch_MNIST(path, valid_size=0.2, seed=None):

    path = Path(path)
    train_path = path / 'mnist.scale.bz2'
    test_path = path / 'mnist.scale.t.bz2'

    if not train_path.exists() or not test_path.exists():
        path.mkdir(parents=True)

        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2', train_path)
        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.t.bz2', test_path)

    X_train, y_train = read_idx_file(train_path, 784, " ", True)
    X_test, y_test = read_idx_file(test_path, 784, " ", True)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size, random_state=seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_PENDIGITS(path, valid_size=0.2, seed=None):

    path = Path(path)
    train_path = path / 'pendigits.data'
    test_path = path / 'pendigits.t.data'

    if not train_path.exists() or not test_path.exists():
        path.mkdir(parents=True)

        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits', train_path)
        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/pendigits.t', test_path)

    X_train, y_train = read_idx_file(train_path, 16, " ")
    X_test, y_test = read_idx_file(test_path, 16, " ")

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size, random_state=seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_PROTEIN(path, valid_size=0.2, seed=None):

    path = Path(path)
    train_path = path / 'protein.bz2'
    test_path = path / 'protein.t.bz2'

    if not train_path.exists() or not test_path.exists():
        path.mkdir(parents=True)

        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/protein.bz2', train_path)
        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/protein.t.bz2', test_path)

    X_train, y_train = read_idx_file(train_path, 357, '  ', True)
    X_test, y_test = read_idx_file(test_path, 357, '  ', True)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size, random_state=seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_SENSORLESS(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    data_path = path / 'sensorless.data'

    if not data_path.exists():
        path.mkdir(parents=True, exist_ok=True)

        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/Sensorless', data_path)

    X, Y = read_idx_file(data_path, 48)
    Y -= 1
    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size / (1 - test_size), random_state=seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_SHUTTLE(path, valid_size=0.2, seed=None):

    path = Path(path)
    train_path = path / 'shuttle.data'
    test_path = path / 'shuttle.t.data'

    if not train_path.exists() or not test_path.exists():
        path.mkdir(parents=True)

        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/shuttle.scale', train_path)
        download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/shuttle.scale.t', test_path)

    X_train, y_train = read_idx_file(train_path, 9, " ")
    y_train -= 1
    X_test, y_test = read_idx_file(test_path, 9, " ")
    y_test -= 1

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size, random_state=seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_FASHION_MNIST(path, valid_size=0.2, seed=None):
    """code adapted from https://github.com/StephanLorenzen/MajorityVoteBounds/blob/278a2811774e48093a7593e068e5958832cfa686/mvb/data.py#L143"""
    path = Path(path)
    train_path = path / 'fashion-mnist-train.data.gz'
    train_label_path = path / 'fashion-mnist-train.label.gz'
    test_path = path / 'fashion-mnist-test.data.gz'
    test_label_path = path / 'fashion-mnist-test.label.gz'

    if not train_path.exists() or not test_path.exists():
        path.mkdir(parents=True)

        download('https://github.com/zalandoresearch/fashion-mnist/blob/master/data/fashion/t10k-images-idx3-ubyte.gz?raw=true', test_path)
        download('https://github.com/zalandoresearch/fashion-mnist/blob/master/data/fashion/t10k-labels-idx1-ubyte.gz?raw=true', test_label_path)
        download('https://github.com/zalandoresearch/fashion-mnist/blob/master/data/fashion/train-images-idx3-ubyte.gz?raw=true', train_path)
        download('https://github.com/zalandoresearch/fashion-mnist/blob/master/data/fashion/train-labels-idx1-ubyte.gz?raw=true', train_label_path)

    with gzip.open(train_path) as f:
        f.read(16)
        buf = f.read(28*28*60000)
        X_train = np.frombuffer(buf, dtype=np.uint8).astype(np.float32).reshape(60000, 28*28)  

    with gzip.open(test_path) as f:
        f.read(16)
        buf = f.read(28*28*10000)
        X_test = np.frombuffer(buf, dtype=np.uint8).astype(np.float32).reshape(10000, 28*28)
    
    with gzip.open(train_label_path) as f:
        f.read(8)
        buf = f.read(60000)
        y_train = np.frombuffer(buf, dtype=np.uint8).astype(np.float32) 

    with gzip.open(test_label_path) as f:
        f.read(8)
        buf = f.read(10000)
        y_test = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size, random_state=seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

# def fetch_USPS(path, valid_size=0.2, seed=None):

#     path = Path(path)
#     train_path = path / 'usps.bz2'
#     test_path = path / 'usps.t.bz2'

#     if not train_path.exists() or not test_path.exists():
#         path.mkdir(parents=True)

#         download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.bz2 ', train_path)
#         download('https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/usps.t.bz2', test_path)

#     import pdb; pdb.set_trace()
    
#     X_train, y_train = read_idx_file(train_path, 256, " ", True)
#     X_test, y_test = read_idx_file(test_path, 256, " ", True)

#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size, random_state=seed)

#     return dict(
#         X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
#     )
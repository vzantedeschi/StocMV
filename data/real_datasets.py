import numpy as np
import gzip
import shutil
import tarfile
import os
import bz2
import pandas as pd

import warnings

from pathlib import Path

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split

from category_encoders import LeaveOneOutEncoder
from category_encoders.ordinal import OrdinalEncoder

from core.utils import download

def fetch_GLASS(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    data_path = path / 'glass.data'

    if not data_path.exists():
        path.mkdir(parents=True)

        download('https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data', data_path)

    data = np.genfromtxt(data_path, delimiter=',')
    
    X, Y = (data[:, 1:-1]).astype(np.float32), (data[:, -1] - 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size / (1 - test_size), random_state=seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_COVTYPE(path, valid_size=0.2, test_size=0.2, seed=None):

    path = Path(path)
    data_path = path / 'covtype.data'

    if not data_path.exists():
        path.mkdir(parents=True)
        archive_path = path / 'covtype.data.gz'
        download('https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/covtype.data.gz', archive_path)

        with gzip.open(archive_path, 'rb') as f_in:

            with open(data_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    data = np.genfromtxt(data_path, delimiter=',')
    
    X, Y = (data[:, :-1]).astype(np.float32), (data[:, -1] - 1).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size / (1 - test_size), random_state=seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_DIGITS(path, valid_size=0.2, test_size=0.2, seed=None):

    from sklearn.datasets import load_digits

    X, Y = load_digits(return_X_y=True)
    X, Y = X.reshape(-1, 1, 8, 8).astype(np.uint8), Y.astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, stratify=Y, test_size=test_size, random_state=seed)

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size / (1 - test_size), random_state=seed)

    return dict(
        X_train=X_train, y_train=y_train, X_valid=X_val, y_valid=y_val, X_test=X_test, y_test=y_test
    )

def fetch_HIGGS(path, train_size=None, valid_size=None, test_size=5 * 10 ** 5, **kwargs):
    data_path = os.path.join(path, 'higgs.csv')
    if not os.path.exists(data_path):
        os.makedirs(path, exist_ok=True)
        archive_path = os.path.join(path, 'HIGGS.csv.gz')
        download('https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz', archive_path)
        with gzip.open(archive_path, 'rb') as f_in:
            with open(data_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    n_features = 29
    types = {i: (np.float32 if i != 0 else np.int) for i in range(n_features)}
    data = pd.read_csv(data_path, header=None, dtype=types)
    data_train, data_test = data.iloc[:-test_size], data.iloc[-test_size:]

    X_train, y_train = data_train.iloc[:, 1:].values, data_train.iloc[:, 0].values
    X_test, y_test = data_test.iloc[:, 1:].values, data_test.iloc[:, 0].values

    if all(sizes is None for sizes in (train_size, valid_size)):
        train_idx_path = os.path.join(path, 'stratified_train_idx.txt')
        valid_idx_path = os.path.join(path, 'stratified_valid_idx.txt')
        if not all(os.path.exists(fname) for fname in (train_idx_path, valid_idx_path)):
            download("https://www.dropbox.com/s/i2uekmwqnp9r4ix/stratified_train_idx.txt?dl=1", train_idx_path)
            download("https://www.dropbox.com/s/wkbk74orytmb2su/stratified_valid_idx.txt?dl=1", valid_idx_path)
        train_idx = pd.read_csv(train_idx_path, header=None)[0].values
        valid_idx = pd.read_csv(valid_idx_path, header=None)[0].values
    else:
        assert train_size, "please provide either train_size or none of sizes"
        if valid_size is None:
            valid_size = len(X_train) - train_size
            assert valid_size > 0
        if train_size + valid_size > len(X_train):
            warnings.warn('train_size + valid_size = {} exceeds dataset size: {}.'.format(
                train_size + valid_size, len(X_train)), Warning)

        shuffled_indices = np.random.permutation(np.arange(len(X_train)))
        train_idx = shuffled_indices[:train_size]
        valid_idx = shuffled_indices[train_size: train_size + valid_size]

    return dict(
        X_train=X_train[train_idx], y_train=y_train[train_idx],
        X_valid=X_train[valid_idx], y_valid=y_train[valid_idx],
        X_test=X_test, y_test=y_test,
    )

def fetch_CLICK(path, valid_size=100_000, seed=None, **kwargs):
    # based on: https://www.kaggle.com/slamnz/primer-airlines-delay
    csv_path = os.path.join(path, 'click.csv')
    if not os.path.exists(csv_path):
        os.makedirs(path, exist_ok=True)
        download('https://www.dropbox.com/s/w43ylgrl331svqc/click.csv?dl=1', csv_path)

    data = pd.read_csv(csv_path, index_col=0)
    X, y = data.drop(columns=['target']), data['target']
    X_train, X_test = X[:-100_000].copy(), X[-100_000:].copy()
    y_train, y_test = y[:-100_000].copy(), y[-100_000:].copy()

    y_train = (y_train.values.reshape(-1) == 1).astype('int64')
    y_test = (y_test.values.reshape(-1) == 1).astype('int64')

    cat_features = ['url_hash', 'ad_id', 'advertiser_id', 'query_id',
                    'keyword_id', 'title_id', 'description_id', 'user_id']

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=valid_size, random_state=seed)

    cat_encoder = LeaveOneOutEncoder()
    cat_encoder.fit(X_train[cat_features], y_train)
    X_train[cat_features] = cat_encoder.transform(X_train[cat_features])
    X_val[cat_features] = cat_encoder.transform(X_val[cat_features])
    X_test[cat_features] = cat_encoder.transform(X_test[cat_features])
    return dict(
        X_train=X_train.values.astype('float32'), y_train=y_train,
        X_valid=X_val.values.astype('float32'), y_valid=y_val,
        X_test=X_test.values.astype('float32'), y_test=y_test
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
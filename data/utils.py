import numpy as np
import bz2
import requests
import os

from sklearn.model_selection import train_test_split

def get_validation_set(X_train, y_train, valid_size, seed):

    X_val, y_val = None, None
    if valid_size > 0:
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=valid_size, random_state=seed)

    return X_val, y_val

"""
Code adapted from https://github.com/Qwicen/node/blob/master/lib/data.py .

"""

def download(url, filename, delete_if_interrupted=True, chunk_size=4096):
    """ saves file from url to filename with a fancy progressbar """
    try:
        with open(filename, "wb") as f:
            print("Downloading {} > {}".format(url, filename))
            response = requests.get(url, stream=True)
            total_length = response.headers.get('content-length')

            if total_length is None:  # no content length header
                f.write(response.content)
            else:
                total_length = int(total_length)
                with tqdm(total=total_length) as progressbar:
                    for data in response.iter_content(chunk_size=chunk_size):
                        if data:  # filter-out keep-alive chunks
                            f.write(data)
                            progressbar.update(len(data))

    except Exception as e:
        if delete_if_interrupted:
            print("Removing incomplete download {}.".format(filename))
            os.remove(filename)
        raise e
    return filename

"""
Code adapted from https://github.com/StephanLorenzen/MajorityVoteBounds/blob/278a2811774e48093a7593e068e5958832cfa686/mvb/data.py#L20
"""

def read_idx_file(path, d, sep=None, bz2_compressed=False):
    X = []
    Y = []

    if bz2_compressed:
        f = bz2.open(path, mode='rt')
    else:
        f = open(path)

    for l in f:
        x = np.zeros(d)
        l = l.strip().split() if sep is None else l.strip().split(sep)
        Y.append(int(l[0]))
        for pair in l[1:]:
            pair = pair.strip()
            if pair=='':
                continue
            (i,v) = pair.split(":")
            if v=='':
                import pdb; pdb.set_trace()
            x[int(i)-1] = float(v)
        X.append(x)

    f.close()
    return np.array(X),np.array(Y)
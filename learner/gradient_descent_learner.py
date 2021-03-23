#!/usr/bin/env python
import torch
import numpy as np
import logging

from sklearn.base import BaseEstimator, ClassifierMixin
from core.numpy_dataset import NumpyDataset
from voter.majority_vote import MajorityVote
from voter.majority_vote_diff import MajorityVoteDiff

###############################################################################


class GradientDescentLearner(BaseEstimator, ClassifierMixin):

    def __init__(
        self, majority_vote, epoch=10, batch_size=None
    ):

        assert isinstance(epoch, int) and epoch > 0
        self.epoch = epoch
        assert (batch_size is None
                or (isinstance(batch_size, int) and batch_size > 0))
        self.batch_size = batch_size

        self.device = "cuda"
        new_device = torch.device('cpu')
        if(torch.cuda.is_available() and self.device != "cpu"):
            new_device = torch.device(self.device)
        self.device = new_device

        self.mv = majority_vote
        assert isinstance(self.mv, MajorityVote)
        assert self.mv.fitted
        #  assert self.mv.complemented

        self.quasi_uniform = self.mv.quasi_uniform

    def fit(self, X, y):
        # X -> (size, nb_feature)
        # y -> (size, 1)
        assert isinstance(X, np.ndarray) and isinstance(y, np.ndarray)
        assert (len(X.shape) == 2 and len(y.shape) == 2 and
                X.shape[0] == y.shape[0] and
                y.shape[1] == 1 and X.shape[0] > 0)

        y_unique = np.sort(np.unique(y))
        assert y_unique[0] == -1 and y_unique[1] == +1

        SEED = 0
        np.random.seed(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.manual_seed(SEED)

        if(self.quasi_uniform):
            self.mv.switch_complemented()

        self.mv_diff = MajorityVoteDiff(
            self.mv, self.device)
        self.mv_diff.to(self.device)

        data = NumpyDataset({
            "x_train": X,
            "y_train": y})

        if(self.batch_size is None or self.batch_size >= X.shape[0]):
            self.batch_size = len(X)
        loader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size)

        # Computing batch size
        num_batch = int(len(data)/self.batch_size)
        if(len(data) % self.batch_size != 0):
            num_batch += 1

        for epoch in range(self.epoch):

            if(self.batch_size != len(X)):
                logging.info(("Running epoch [{}/{}] ...\n").format(
                    epoch+1, self.epoch))

            loss_sum = 0.0

            for i, batch in enumerate(loader):

                batch["m"] = len(X)
                batch["x"] = batch["x"].to(
                    device=self.device, dtype=torch.float32)
                batch["y"] = batch["y"].to(
                    device=self.device, dtype=torch.long)

                # Optimize the model
                # REQUIRED:
                #   (1) create self._loss
                #   (2) create the dict self._log
                self._loss = None
                self._log = {}
                self._optimize(batch)
                assert self._loss is not None

                # Computing mean loss
                loss_sum += self._loss
                loss_mean = loss_sum/(i+1)

                if(self.batch_size != len(X)):
                    loading = (i+1, num_batch)
                else:
                    loading = (epoch+1, self.epoch)

                # Printing loss and criteria
                logging_str = "[{}/{}] - loss {:.4f}".format(
                        loading[0], loading[1], loss_mean)
                for key, value in self._log.items():
                    logging_str += self.__print_logging(key, value)
                logging.info(logging_str+"\r")
                #  logging.info(logging_str+"\n")

                if(self.batch_size != len(X)):
                    if i+1 == num_batch:
                        logging.info("\n")

        if(self.batch_size == len(X)):
            logging.info("\n")

        self.mv.post = self.mv_diff.post.cpu().detach().numpy()

        if(self.quasi_uniform):
            self.mv.switch_complemented()
            # Note: In this case, self.mv will be complemented
            # unlike self.mv_diff

        return self

    def __print_logging(self, key, value):
        if(isinstance(value, int)):
            return " - {} {}".format(key, value)
        elif(isinstance(value, float)):
            return " - {} {:.4f}".format(key, value)
        elif(isinstance(value, str)):
            return " - {} {}".format(key, value)
        elif(isinstance(value, torch.Tensor)):
            return self.__print_logging(key, value.cpu().detach().numpy())
        elif(isinstance(value, np.ndarray)):
            if(value.ndim == 0):
                return self.__print_logging(key, value.item())
            else:
                raise ValueError("value cannot be an array")
        else:
            raise TypeError(
                "value must be of type torch.Tensor; np.ndarray;"
                + " int; float or str.")

    def predict(self, X):
        data = NumpyDataset({"x_test": X})
        data.set_mode("test")
        if(self.batch_size is None):
            self.batch_size = len(X)
        loader = torch.utils.data.DataLoader(
            data, batch_size=self.batch_size)

        if(self.quasi_uniform):
            self.mv.switch_complemented()

        pred = None
        for i, batch in enumerate(loader):
            batch["m"] = len(X)
            batch["x"] = batch["x"].to(
                device=self.device, dtype=torch.float32)

            self.mv_diff(batch)
            pred_ = self.mv_diff.pred.detach().numpy()

            if(pred is None):
                pred = pred_
            else:
                pred = np.concatenate((pred, pred_))

        if(self.quasi_uniform):
            self.mv.switch_complemented()

        return pred

    def _optimize(self):
        raise NotImplementedError


###############################################################################

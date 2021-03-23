from h5py import File
from re import sub
import re
from os.path import isfile
from torch.utils.data import Dataset
import torch
import numpy as np
import copy


class NumpyDataset(Dataset):

    def __init__(self, x_y_dict):

        self._dataset_key = list(x_y_dict.keys())

        self._mode_list = []

        # Removing the mode in self._dataset_key
        new_dataset_key = {}
        for key in self._dataset_key:
            new_key = sub("_[^_]+$", "", key)
            if(new_key != key):
                new_dataset_key[new_key] = None
        self._dataset_key = list(new_dataset_key.keys())

        self._dataset_dict = copy.deepcopy(x_y_dict)
        self._mode = "train"

    def set_mode(self, mode):
        # Setting the mode of the dataset
        self._mode = mode

    def get_mode(self):
        # Getting the mode of the dataset
        return self._mode

    def get_mode_dataset(self, key):
        if(self._mode == "train" or self._mode == "test"):
            mode_key = key+"_"+self._mode
            mode_dict_key = key+"_"+self._mode
        else:
            mode_key = key+"_train"
            mode_dict_key = key+"_train_"+self._mode

        if(mode_dict_key in self._dataset_dict):
            return self._dataset_dict[mode_dict_key]

        #  if(self._mode == "train" or self._mode == "test"):
        #      self._dataset_dict[mode_dict_key] = self._dataset[mode_key]
        #  else:
        #      mode_index = self._mode_list.index(self._mode)
        #
        #      if(mode_index == 0):
        #          end_index = self._size_list[mode_index]
        #
        #          self._dataset_dict[mode_dict_key] = (
        #              self._dataset[mode_key][:end_index])
        #
        #          return self._dataset[mode_key][:end_index]
        #      else:
        #          begin_index = self._size_list[mode_index-1]
        #          end_index = self._size_list[mode_index]
        #          self._dataset_dict[mode_dict_key] = (
        #              self._dataset[mode_key][begin_index:end_index])

        return self._dataset_dict[mode_dict_key]

    def __len__(self):
        # Getting the size of a dataset (of a given "mode")
        return len(self.get_mode_dataset("x"))

    def class_size(self):
        if("y"+self._mode in self._dataset_dict):
            return len(np.unique(self.get_mode_dataset("y")))
        return 1

    def input_size(self):
        return list(self.get_mode_dataset("x").shape[1:])

    def __getitem__(self, i):
        # Getting each example for a given mode
        item_dict = {
            "mode": self._mode,
            "size": self.__len__(),
            "class_size": self.class_size()
        }
        for key in self._dataset_key:

            # If we have an example, we transform the example before
            if(key == "x"):
                example = torch.tensor(self.get_mode_dataset(key)[i])
                #  if(self._mode in self._transform_dict):
                    #  example = self._transform_dict[self._mode](example)
                #  else:
                    #  example = self._transform_dict["default"](example)

                item_dict[key] = example
            else:
                item_dict[key] = torch.tensor(self.get_mode_dataset(key)[i])

        return item_dict

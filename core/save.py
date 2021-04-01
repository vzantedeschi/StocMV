import pandas as pd
import numpy as np


def save_csv(name_csv, dict_save, name_exp, erase=False):
    try:
        csv_pd = pd.read_csv(name_csv, index_col=0)
    except FileNotFoundError:
        csv_pd = pd.DataFrame()

    if(name_exp in csv_pd.index):
        for column, value in dict_save.items():
            if(column not in csv_pd.columns):
                csv_pd[column] = np.nan

            if(np.isnan(csv_pd.loc[name_exp][column])
               or (not(np.isnan(csv_pd.loc[name_exp][column])) and erase)):
                csv_pd.at[name_exp, column] = value

        csv_pd.to_csv(name_csv)
        return None

    old_set = set(csv_pd.columns)
    new_set = set(dict_save.keys())

    new_column = old_set.difference(new_set)
    new_column = new_column.union(new_set.difference(old_set))
    new_column = sorted(list(new_column))

    for column in new_column:
        csv_pd[column] = np.nan

    csv_pd.loc[name_exp] = dict_save
    csv_pd.to_csv(name_csv)


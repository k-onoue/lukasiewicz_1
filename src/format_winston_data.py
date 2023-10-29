import os

import numpy as np
import pandas as pd


class WinstonDataFormatter:

    def __init__(self, df_data, df_labels):

        self.target_animal_names = ['albatross', 'cheetah', 'ostrich', 'penguin', 'zebra']

        self.df_data = df_data
        self.df_labels = df_labels

    def format_data(self):
        self.data_dict = {}

        for predicate_name, row in self.df_labels.iterrows():
            df_tmp = self.df_data.loc[:, ['folder', 'R', 'G', 'B']].copy()
            df_tmp.loc[:, 'label'] = 'hello'

            for folder_name in self.target_animal_names:
                condition = df_tmp.loc[:, 'folder'] == folder_name
                df_tmp.loc[condition, 'label'] = row[folder_name]

            df_tmp = df_tmp.drop(['folder'], axis=1)

            self.data_dict[predicate_name] = df_tmp

    def save_as_csv(self, save_dir_path):
        for predicate_name, data in self.data_dict.items():
            file_name = 'L_' + predicate_name + '.csv'
            data.to_csv(os.path.join(save_dir_path, file_name))



def generate_and_save_unsupervised_data(save_dir_path, data_num=10, data_dim=3):
    arr_U = np.random.rand(data_num, data_dim)
    df_U = pd.DataFrame(arr_U)
    df_U.to_csv(os.path.join(save_dir_path, 'U.csv'))
    print('Done!')
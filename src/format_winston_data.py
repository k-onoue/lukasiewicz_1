import os

import numpy as np
import pandas as pd



class WinstonDataFormatter:

    def __init__(self, source_dir_path):

        self.mapping_dict = {
            'n02058221': 'albatross',
            'n02130308': 'cheetah',
            'n01518878': 'ostrich',
            'n02056570': 'penguin',
            # 'n02391049': 'zebra'
        }

        self.source_dir_path = source_dir_path

        self.df_labels = self._load_labels_df()
        self.df_data = self._load_data_df()

    def _load_labels_df(self):
        file_path = os.path.join(self.source_dir_path, 'labels.csv')
        df_labels = pd.read_csv(file_path, index_col=0).fillna(-1)
        df_labels.columns = self.mapping_dict.keys()
        
        return df_labels

    def _load_data_df(self):
        file_path = os.path.join(self.source_dir_path, 'data.csv')
        df_data = pd.read_csv(file_path, index_col=0)
        return df_data

    def format_and_save_data(self, 
                             save_dir_path, 
                             sample_num_per_animal=None):

        for idx, row in self.df_labels.iterrows():
            file_name = "L_" + idx + ".csv"
            df_tmp = self.df_data.loc[:, ['folder', 'R', 'G', 'B']].copy()
            df_tmp.loc[:, 'label'] = 'hello'

            for folder_name in self.mapping_dict.keys():
                condition = df_tmp.loc[:, 'folder'] == folder_name
                df_tmp.loc[condition, 'label'] = row[folder_name]

                if sample_num_per_animal is None:
                    condition = df_tmp.loc[:, 'folder'] == folder_name
                    df_tmp.loc[condition, 'label'] = row[folder_name]
                elif type(sample_num_per_animal) == int and sample_num_per_animal > 0:
                    selected_rows = df_tmp[condition].sample(n=sample_num_per_animal)
                    df_tmp = pd.concat([selected_rows, df_tmp[~condition]])
                else:
                    print("Something wrong!")
                    break

            df_tmp = df_tmp.drop(['folder'], axis=1)

            df_tmp.to_csv(os.path.join(save_dir_path, file_name))

        print('Done!')  
    

def generate_and_save_unsupervised_data(save_dir_path, data_num=10, data_dim=3):
    arr_U = np.random.rand(data_num, data_dim)
    df_U = pd.DataFrame(arr_U)
    df_U.to_csv(os.path.join(save_dir_path, 'U.csv'))
    print('Done!')
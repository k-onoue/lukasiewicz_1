import os
import shutil

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold


setting_dict = {
    'seed': 42,
    'n_splits': 5, # 'test_size' の代わり
    'source_path': 'data/pima_indian_diabetes',
    'source_data_file_name': 'diabetes_discretized.csv',
    'source_rule_file_name': 'rule.txt', ##########
    'input_path': 'inputs/pima_indian_diabetes_cv_2',
    'unsupervised_file_name': 'U.csv',
    'unsupervised_shape': (100, 21), # (data_num, data_dim) ########################
    'output_path': 'outputs/pima_indian_diabetes_8' # この辺の命名規則 （input_path が _2 になっている）はわかりにくいので見直す
}


def prepare_data(setting: dict) -> None:
    random_state = setting['seed']
    n_splits = setting['n_splits']

    source_data_path = os.path.join(setting['source_path'], setting['source_data_file_name'])
    input_path = setting['input_path']

    data = pd.read_csv(source_data_path, index_col=0)
    
    data = data.reset_index(drop=True)

    X = data.drop(["Outcome"], axis=1)
    y = data["Outcome"]

    kf = KFold(n_splits=n_splits)
    
    for i, (train_index, test_index) in enumerate(kf.split(X)):

        train_data = data.loc[train_index, :]
        outcome = train_data['Outcome']
        features = train_data.drop(['Outcome'], axis=1)
        feature_names = list(features.columns)

        input_train_path = os.path.join(input_path, f'fold_{i}', 'train')

        if not os.path.exists(input_train_path):
            os.makedirs(input_train_path)

        df = features.copy()
        df['target'] = outcome.replace(0, -1)

        file_name = "L_" + "Outcome" + '.csv'
        file_path = os.path.join(input_train_path, file_name)
        df.to_csv(file_path)

        for feature_name in feature_names:
            df = features.copy()
            df['target'] = df[feature_name].replace(0, -1)

            file_name = "L_" + feature_name + '.csv'
            file_path = os.path.join(input_train_path, file_name)
            df.to_csv(file_path)

        unsupervised_path = os.path.join(input_train_path, setting['unsupervised_file_name'])
        unsupervised_shape = setting['unsupervised_shape']

        arr_U = np.random.randint(2, size=unsupervised_shape)
        df_U = pd.DataFrame(arr_U)
        df_U.to_csv(unsupervised_path)

        rule_file_name = setting['source_rule_file_name']
        source_rule_path = os.path.join(setting['source_path'], rule_file_name)
        rule_path = os.path.join(input_train_path, rule_file_name)

        shutil.copy(source_rule_path, rule_path)

        test_data = data.loc[test_index, :]

        outcome = test_data['Outcome']
        features = test_data.drop(['Outcome'], axis=1)
        feature_names = list(features.columns)

        input_test_path = os.path.join(input_path, f'fold_{i}', 'test')

        if not os.path.exists(input_test_path):
            os.makedirs(input_test_path)

        
        df = features.copy()
        df['target'] = outcome.replace(0, -1)

        file_name = "L_" + "Outcome" + '.csv'
        file_path = os.path.join(input_test_path, file_name)
        df.to_csv(file_path)


if __name__ == '__main__':
    prepare_data(setting_dict)
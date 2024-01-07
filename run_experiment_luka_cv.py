"""
Pima Indian Diabetes データセット用の実験を行うためのファイルです

setting_dict と objectives_dict に実験の設定を記述し、

実験結果を json ファイルとして保存します。

cross-validation ができるようになりました。
"""

import os
import shutil

import pandas as pd
import numpy as np
import cvxpy as cp
from sklearn.model_selection import KFold

from src.setup_problem_primal_modular import Setup
from src.objective_function import linear_svm
from src.objective_function import linear_svm_loss
from src.objective_function import logistic_regression_loss
from src.evaluation import EvaluateModel


setting_dict = {
    'seed': 42,
    'n_splits': 5, # 'test_size' の代わり
    'source_path': 'data/pima_indian_diabetes',
    'source_data_file_name': 'diabetes_discretized.csv',
    'source_rule_file_name': 'rules_2.txt', ##########
    'input_path': 'inputs/pima_indian_diabetes_cv',
    'unsupervised_file_name': 'U.csv',
    'unsupervised_shape': (15, 21), # (data_num, data_dim)
    'output_path': 'outputs/pima_indian_diabetes_3'
}

objectives_dict = {
    'luka_1': {
        'model_name': 'luka linear svm',
        'model': linear_svm,
        'params': {'c1': 10, 'c2': 10},
        'constraints_flag': {
            'pointwise': True,
            'logical': True,
            'consistency': True
        }
    },
    'luka_2': {
        'model_name': 'luka linear svm loss',
        'model': linear_svm_loss,
        'params': {'c1': 10, 'c2': 10},
        'constraints_flag': {
            'pointwise': False,
            'logical': False,
            'consistency': True
        }
    },
    'luka_3': {
        'model_name': 'luka logistic regression loss',
        'model': logistic_regression_loss,
        'params': {'c1': 10, 'c2': 10},
        'constraints_flag': {
            'pointwise': False,
            'logical': False,
            'consistency': True
        }
    },
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

    for i in range(setting_dict['n_splits']):
        data_dir_path = os.path.join(setting_dict['input_path'], f'fold_{i}')
        
        file_list = os.listdir(os.path.join(data_dir_path, "train"))

        L_files = [filename for filename in file_list 
                if filename.startswith('L') and filename.endswith('.csv')]

        U_files = [filename for filename in file_list 
                if filename.startswith('U') and filename.endswith('.csv')]

        file_names_dict = {
            'supervised': L_files,
            'unsupervised': U_files,
            'rule': [setting_dict['source_rule_file_name']]
        }

        for key, obj_setting in objectives_dict.items():

            model_name = obj_setting['model_name']
            obj_constructor = obj_setting['model']
            c1 = obj_setting['params']['c1']
            c2 = obj_setting['params']['c2']
            constraints_flag = obj_setting['constraints_flag']


            print()
            print(f'model name: {model_name}')
            print(f'obj constructor: {obj_constructor}')
            print(f'c1: {c1}')
            print(f'c2: {c2}')
            print()

            problem_instance = Setup(data_dir_path,
                                    file_names_dict,
                                    obj_constructor,
                                    name=model_name)
            
            objective_function, constraints = problem_instance.main(c1=c1, c2=c2,
                                                                    constraints_flag_dict=constraints_flag)
            
            problem = cp.Problem(objective_function, constraints)
            result = problem.solve(verbose=True)

            print(result)

            save_file_name = 'result_' + key + '.json'
            save_dir_path = os.path.join(setting_dict['output_path'], f'fold_{i}')
            if not os.path.exists(save_dir_path):
                os.makedirs(save_dir_path)
            save_file_path = os.path.join(save_dir_path, save_file_name)
            note = None

            evaluate_model = EvaluateModel(problem_instance, note=note)
            evaluate_model.evaluate(save_file_path=save_file_path)
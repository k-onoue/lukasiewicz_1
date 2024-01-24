"""
Pima Indian Diabetes データセット用の実験を行うためのファイルです

setting_dict と objectives_dict に実験の設定を記述し、

実験結果を json ファイルとして保存します。
"""

import os
import shutil

import pandas as pd
import numpy as np
import cvxpy as cp
from sklearn.model_selection import train_test_split

from src.setup_problem_primal_modular import Setup
from src.objective_function import linear_svm
from src.objective_function import linear_svm_loss
from src.objective_function import logistic_regression_loss
from src.evaluation_conti import EvaluateModel


setting_dict = {
    'seed': 42,
    'test_size': 0.2,
    'source_path': 'data/pima_indian_diabetes',
    'source_data_file_name': 'diabetes_discretized.csv',
    'source_data_file_name_2': 'diabetes_cleaned_standardized.csv',
    'source_rule_file_name': 'rules_3.txt',
    'input_path': 'inputs/pima_indian_diabetes_3',
    'unsupervised_file_name': 'U.csv',
    'unsupervised_shape': (15, 7), # (data_num, data_dim)
    'output_path': 'outputs/pima_indian_diabetes_7'
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
    test_size = setting['test_size']
    source_data_path = os.path.join(setting['source_path'], setting['source_data_file_name'])
    source_data_path_2 = os.path.join(setting['source_path'], setting['source_data_file_name_2'])
    input_path = setting['input_path']
    input_train_path = os.path.join(input_path, 'train')
    input_test_path = os.path.join(input_path, 'test')
    rule_file_name = setting['source_rule_file_name']
    source_rule_path = os.path.join(setting['source_path'], rule_file_name)
    rule_path = os.path.join(input_train_path, rule_file_name)
    unsupervised_path = os.path.join(input_train_path, setting['unsupervised_file_name'])
    unsupervised_shape = setting['unsupervised_shape']

    data = pd.read_csv(source_data_path, index_col=0)
    X = data.drop(["Outcome"], axis=1)
    y = data["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=test_size,
                                                        random_state=random_state)

    train_index = X_train.index
    test_index = X_test.index

    ##############################################################
    data_conti = pd.read_csv(source_data_path_2, index_col=0)

    data_conti_train = data_conti.loc[train_index, :]
    data_conti_test  = data_conti.loc[test_index, :]
    features_conti_train = data_conti_train.drop(["Outcome"], axis=1)
    ##############################################################

    train_data = data.loc[train_index, :]
    outcome = train_data['Outcome']
    features = train_data.drop(['Outcome'], axis=1)
    feature_names = list(features.columns)

    if not os.path.exists(input_train_path):
        os.makedirs(input_train_path)

    df = features_conti_train.copy()
    df['target'] = outcome.replace(0, -1)

    file_name = "L_" + "Outcome" + '.csv'
    file_path = os.path.join(input_train_path, file_name)
    df.to_csv(file_path)

    for feature_name in feature_names:
        df = features_conti_train.copy()
        df['target'] = features[feature_name].replace(0, -1)
        # display(df)

        file_name = "L_" + feature_name + '.csv'
        file_path = os.path.join(input_train_path, file_name)
        df.to_csv(file_path)


    features_conti_train = features_conti_train.copy()
    min_values = features_conti_train.min()
    max_values = features_conti_train.max()
    num_samples = unsupervised_shape[0]
    dict_U = {}
    for column in features_conti_train.columns:
        dict_U[column] = np.random.uniform(min_values[column], max_values[column], num_samples)
    df_U = pd.DataFrame(dict_U)
    df_U.to_csv(unsupervised_path)


    shutil.copy(source_rule_path, rule_path)


    if not os.path.exists(input_test_path):
        os.makedirs(input_test_path)

    df = data_conti_test.copy()
    df.rename(columns={'Outcome': 'target'}, inplace=True)
    df['target'] = df['target'].replace(0, -1)

    file_name = "L_" + "Outcome" + '.csv'
    file_path = os.path.join(input_test_path, file_name)
    df.to_csv(file_path)



if __name__ == '__main__':
    coeff_check = []



    prepare_data(setting_dict)

    data_dir_path = setting_dict['input_path']
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
        print()
        print()
        print(model_name)
        print(obj_constructor)
        print(c1)
        print(c2)
        print(constraints_flag)
        print()
        print()
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
        print()
        print()
        print()

        coeff_check.append(problem_instance.predicates_dict['Outcome'].w.value)

        save_file_name = 'result_' + key + '.json'
        save_file_path = os.path.join(setting_dict['output_path'], save_file_name)
        note = None


        path_discretized = os.path.join(setting_dict['source_path'], setting_dict['source_data_file_name'])
        test_size = setting_dict['test_size']
        random_state = setting_dict['seed']

        evaluate_model = EvaluateModel(problem_instance, 
                                       path_discretized, 
                                       test_size,
                                       random_state,
                                       note=note)
        evaluate_model.evaluate(save_file_path=save_file_path)

    print()
    print()
    print()
    print()
    print()
    print("coeff: ")
    print(coeff_check)
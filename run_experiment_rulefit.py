# """
# 論文の提案アルゴリズムとの比較のために RuleFit を同じデータに対して適用し、評価します

# 実験結果を json ファイルとして保存します。
# """

# import os
# import shutil
# from typing import List
# import json

# import pandas as pd
# import numpy as np
# import cvxpy as cp
# from sklearn.model_selection import train_test_split
# from sklearn.svm import SVC
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score

# from src.misc import is_symbol
# from src.setup_problem_primal_modular import Setup




# setting_dict = {
#     'seed': 42,
#     'test_size': 0.2,
#     'source_path': 'data/pima_indian_diabetes',
#     'source_data_file_name': 'diabetes_discretized.csv',
#     'source_data_file_name_2': 'diabetes_cleaned.csv',
#     'source_rule_file_name': 'rules.txt',
#     'input_path': 'inputs/pima_indian_diabetes',
#     'unsupervised_file_name': 'U.csv',
#     'unsupervised_shape': (15, 21), # (data_num, data_dim)
#     'output_path': 'outputs/pima_indian_diabetes'
# }


# objectives_dict = {
#     'rulefit_1': {
#         'model_name': 'random forest (rulefit)',
#         'model': SVC(kernel='linear', 
#                      random_state=setting_dict['seed'],
#                      probability=True),
#     },
#     'rulefit_2': {
#         'model_name': 'non-linear svm (rbf)',
#         'model': SVC(kernel='rbf',
#                      random_state=setting_dict['seed'],
#                      probability=True),
#     },
# }

# class EvaluateModel:
#     def __init__(self,
#                  path_cleaned: str,
#                  path_discretized: str,
#                  model: object,
#                  KB_origin: List[List[str]],
#                  random_state: int,
#                  test_size: float,
#                  name: str = None,
#                  note: str = None) -> None:

#         self.path_cleaned = path_cleaned
#         self.path_discretized = path_discretized
#         self.model = model
#         self.KB_origin = KB_origin
#         self.random_state = random_state 
#         self.test_size = test_size

#         self.result_dict = {
#             'name'     : name,
#             'note'     : note,
#             'Accuracy' : None,
#             'Precision': None,
#             'Recall'   : None,
#             'F1-score' : None,
#             'Auc'      : None,
#             'len_U': None,
#             'Rules'    : {'violation': 0, 'total': len(self.KB_origin)},
#             'Rules_detail': {}
#         }

#     def calculate_scores(self) -> None:
#         # まずは連続データで計算
#         data = pd.read_csv(self.path_cleaned, index_col=0)
#         X = data.drop('Outcome', axis=1)

#         #####################################################3
#         y = data['Outcome']
#         y.replace(0, -1, inplace=True)

#         X_train, X_test, y_train, y_test = train_test_split(X, y, 
#                                                             test_size=self.test_size, 
#                                                             random_state=self.random_state)

#         idx_tmp = X_test.index

#         feature_names = list(X_train.columns)
#         X_train = X_train.values
#         X_test  = X_test.values
#         y_train = y_train.values
#         y_test  = y_test.values        
#         self.model.fit(X_train, y_train, feature_names=feature_names)


#         y_pred = self.model.tree_generator.predict(X_test)

#         y_pred_interpreted = np.where(y_pred == 0, -1, y_pred)
        
#         y_pred = self.model.tree_generator.predict_proba(X_test)[:, 1]

#         # 精度等の一般的な評価指標の計算
#         accuracy = accuracy_score(y_test, y_pred_interpreted)
#         precision = precision_score(y_test, y_pred_interpreted)
#         recall = recall_score(y_test, y_pred_interpreted)
#         f1 = f1_score(y_test, y_pred_interpreted)
#         roc_auc = roc_auc_score(y_test, y_pred)

#         self.result_dict['Accuracy'] = float(accuracy)
#         self.result_dict['Precision'] = float(precision)
#         self.result_dict['Recall'] = float(recall)
#         self.result_dict['F1-score'] = float(f1)
#         self.result_dict['Auc'] = float(roc_auc)

#         # ルール違反の計算の前に X_test を離散化（discretized のほうを読み込む）
#         data = pd.read_csv(self.path_discretized, index_col=0)
#         X = data.drop('Outcome', axis=1)
#         y = data['Outcome']
#         y.replace(0, -1, inplace=True)
#         X_train, X_test, y_train, y_test = train_test_split(X, y, 
#                                                             test_size=self.test_size, 
#                                                             random_state=self.random_state)
        
#         # ルール違反
#         rules_tmp = []
#         for rule in self.KB_origin:
#             if "Outcome" in rule:
#                 tmp = {}
#                 for idx, item in enumerate(rule):
#                     if not is_symbol(item):
#                         if idx == 0 or rule[idx - 1] != '¬':
#                             tmp[item] = 1
#                         elif item != "Outcome":
#                             tmp[item] = 0
#                         else:
#                             tmp[item] = -1
#                 rules_tmp.append(tmp)

#         idx_tmp = X_test.index
#         y_pred_interpreted = pd.DataFrame(y_pred_interpreted, index=idx_tmp)

#         for i, rule in enumerate(rules_tmp):
#             outcome = rule["Outcome"]
#             condition = " & ".join([f"{column} == {value}" for column, value in rule.items() if column != "Outcome"])

#             tmp = y_pred_interpreted.loc[X_test.query(condition).index]

#             violation_bool = 1 if int((tmp != outcome).sum().iloc[0]) >= 1 else 0
#             self.result_dict['Rules']['violation'] += violation_bool
#             self.result_dict['Rules_detail'][i] = {
#                 'rule': " ".join(self.KB_origin[i]),
#                 'violation': violation_bool,
#             }

#     def save_result_as_json(self, file_path) -> None:
#         with open(file_path, 'w') as f:
#             json.dump(self.result_dict, f, indent=4)

#     def evaluate(self, save_file_path: str = './result_1.json') -> None:
#         self.calculate_scores()
#         self.save_result_as_json(file_path=save_file_path)


# class EvaluateModelRuleFit2:
#     def __init__(self,
#                  path_discretized: str,
#                  model: object,
#                  KB_origin: List[List[str]],
#                  random_state: int,
#                  test_size: float,
#                  name: str = None,
#                  note: str = None) -> None:

#         self.path_discretized = path_discretized
#         self.model = model
#         self.KB_origin = KB_origin
#         self.random_state = random_state 
#         self.test_size = test_size

#         self.result_dict = {
#             'name'     : name,
#             'note'     : note,
#             'Accuracy' : None,
#             'Precision': None,
#             'Recall'   : None,
#             'F1-score' : None,
#             'Auc'      : None,
#             'len_U': None,
#             'Rules'    : {'violation': 0, 'total': len(self.KB_origin)},
#             'Rules_detail': {}
#         }

#     def calculate_scores(self) -> None:
#         data = pd.read_csv(self.path_discretized, index_col=0)
#         X = data.drop('Outcome', axis=1)

#         #####################################################3
#         y = data['Outcome']
#         y.replace(0, -1, inplace=True)

#         X_train, X_test, y_train, y_test = train_test_split(X, y, 
#                                                             test_size=self.test_size, 
#                                                             random_state=self.random_state)
        

#         idx_tmp = X_test.index

#         feature_names = list(X_train.columns)
#         X_train = X_train.values
#         X_test  = X_test.values
#         y_train = y_train.values
#         y_test  = y_test.values        
#         self.model.fit(X_train, y_train, feature_names=feature_names)


#         y_pred = self.model.predict(X_test)

#         y_pred_interpreted = np.where(y_pred == 0, -1, y_pred)
        
#         y_pred = self.model.predict_proba(X_test)[:, 1]

#         # 精度等の一般的な評価指標の計算
#         accuracy = accuracy_score(y_test, y_pred_interpreted)
#         precision = precision_score(y_test, y_pred_interpreted)
#         recall = recall_score(y_test, y_pred_interpreted)
#         f1 = f1_score(y_test, y_pred_interpreted)
#         roc_auc = roc_auc_score(y_test, y_pred)

#         self.result_dict['Accuracy'] = float(accuracy)
#         self.result_dict['Precision'] = float(precision)
#         self.result_dict['Recall'] = float(recall)
#         self.result_dict['F1-score'] = float(f1)
#         self.result_dict['Auc'] = float(roc_auc)

#         # ルール違反
#         rules_tmp = []
#         for rule in self.KB_origin:
#             if "Outcome" in rule:
#                 tmp = {}
#                 for idx, item in enumerate(rule):
#                     if not is_symbol(item):
#                         if idx == 0 or rule[idx - 1] != '¬':
#                             tmp[item] = 1
#                         elif item != "Outcome":
#                             tmp[item] = 0
#                         else:
#                             tmp[item] = -1
#                 rules_tmp.append(tmp)

#         idx_tmp = X_test.index
#         y_pred_interpreted = pd.DataFrame(y_pred_interpreted, index=idx_tmp)

#         for i, rule in enumerate(rules_tmp):
#             outcome = rule["Outcome"]
#             condition = " & ".join([f"{column} == {value}" for column, value in rule.items() if column != "Outcome"])

#             tmp = y_pred_interpreted.loc[X_test.query(condition).index]

#             violation_bool = 1 if int((tmp != outcome).sum().iloc[0]) >= 1 else 0
#             self.result_dict['Rules']['violation'] += violation_bool
#             self.result_dict['Rules_detail'][i] = {
#                 'rule': " ".join(self.KB_origin[i]),
#                 'violation': violation_bool,
#             }

#     def save_result_as_json(self, file_path) -> None:
#         with open(file_path, 'w') as f:
#             json.dump(self.result_dict, f, indent=4)

#     def evaluate(self, save_file_path: str = './result_1.json') -> None:
#         self.calculate_scores()
#         self.save_result_as_json(file_path=save_file_path)




# if __name__ == '__main__':
#     data_dir_path = setting_dict['source_path']
#     path_discretized = os.path.join(data_dir_path, setting_dict['source_data_file_name'])
#     path_cleaned = os.path.join(data_dir_path, setting_dict['source_data_file_name_2'])
#     random_state = setting_dict['seed']
#     test_size = setting_dict['test_size']

#     file_names_dict = {
#         'rule': [setting_dict['source_rule_file_name']]
#     }

#     problem_instance = Setup(setting_dict['input_path'], file_names_dict, None)
#     problem_instance.load_rules()
#     KB_origin = problem_instance.KB_origin

#     note = None


#     for key, obj_setting in objectives_dict.items():

#         model_name = obj_setting['model_name']
#         model = obj_setting['model']

#         save_file_name = 'result_' + key + '.json'
#         save_file_path = os.path.join(setting_dict['output_path'], save_file_name)

#         evaluate_model = EvaluateModel(path_cleaned,
#                                        path_discretized,
#                                        model,
#                                        KB_origin,
#                                        random_state,
#                                        test_size,
#                                        name=model_name, 
#                                        note=note)
#         evaluate_model.evaluate(save_file_path=save_file_path)


# evaluate_model = EvaluateModelRuleFit2(path_discretized=path_discretized,
#                                        model=model,
#                                        name=model_name,
#                                        random_state=random_state,
#                                        test_size=test_size,
#                                        KB_origin=problem_instance.KB_origin)

# evaluate_model.evaluate(save_file_path=save_file_path)
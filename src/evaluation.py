import os
from typing import Dict, Any, List
import json

# from .setup_problem import Setup
class Setup_:
    """
    型ヒント用（circular import の回避のため）
    """
    def __init__(self):
        pass

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
# from sklearn.metrics import confusion_matrix, classification_report

from .misc import is_symbol
from .operators import negation



class EvaluateModel:
    """
    とりあえず，Pima Indian Diabetes のみで使用可能．
    学習済みモデルの評価値を算出する.
    現状 Pima Indian Diabetes dataset にのみ機能する（少しの変更を評価対象の predicate 名に変更すればおそらくうまくいく）．
    または使用するデータセットすべてにおいて，正解ラベルの列の列名を Outcome に変更すると問題なく動くかもしれない．
    """
    def __init__(self,
                 obj: Setup_,
                 note: str = None) -> None:

        self.predicates_dict = obj.predicates_dict
        self.data_dir_path = obj.data_dir_path

        self.KB_origin = obj.KB_origin

        self.result_dict = {
            'name'     : obj.name,
            'note'     : note,
            'Accuracy' : None,
            'Precision': None,
            'Recall'   : None,
            'F1-score' : None,
            'Auc'      : None,
            'len_U': len(next(iter(obj.U.values()))), # size of unsupervised data
            'Rules': {'violation': 0, 'total': len(self.KB_origin)},
            'Rules_detail': {}
        }

    def calculate_scores(self) -> None:
        file_path = os.path.join(self.data_dir_path, "test", "L_Outcome.csv")

        test_data = pd.read_csv(file_path, index_col=0)
        X_test = test_data.drop(['target'], axis=1)
        y_test = test_data['target']

        p = self.predicates_dict['Outcome']
        y_pred = p(X_test).value
        y_pred_interpreted = np.where(y_pred >= 0.5, 1, -1)

        # 精度等の一般的な評価指標の計算
        accuracy = accuracy_score(y_test, y_pred_interpreted)
        precision = precision_score(y_test, y_pred_interpreted)
        recall = recall_score(y_test, y_pred_interpreted)
        f1 = f1_score(y_test, y_pred_interpreted)
        roc_auc = roc_auc_score(y_test, y_pred)

        self.result_dict['Accuracy'] = float(accuracy)
        self.result_dict['Precision'] = float(precision)
        self.result_dict['Recall'] = float(recall)
        self.result_dict['F1-score'] = float(f1)
        self.result_dict['Auc'] = float(roc_auc)

        # ルール違反
        rules_tmp = []
        for rule in self.KB_origin:
            if "Outcome" in rule:
                tmp = {}
                for idx, item in enumerate(rule):
                    if not is_symbol(item):
                        if idx == 0 or rule[idx - 1] != '¬':
                            tmp[item] = 1
                        elif item != "Outcome":
                            tmp[item] = 0
                        else:
                            tmp[item] = -1

                rules_tmp.append(tmp)

        idx_tmp = X_test.index
        y_pred_interpreted = pd.DataFrame(y_pred_interpreted, index=idx_tmp)


        for i, rule in enumerate(rules_tmp):
            outcome = rule["Outcome"]
            condition = " & ".join([f"{column} == {value}" for column, value in rule.items() if column != "Outcome"])
            
            tmp = y_pred_interpreted.loc[X_test.query(condition).index]

            violation_bool = 1 if int((tmp != outcome).sum().iloc[0]) >= 1 else 0
            self.result_dict['Rules']['violation'] += violation_bool
            self.result_dict['Rules_detail'][i] = {
                'rule': " ".join(self.KB_origin[i]),
                'violation': violation_bool, # 1 なら破られた、0 なら守られたルールであることを示す
            }

    def save_result_as_json(self, file_path) -> None:
        with open(file_path, 'w') as f:
            json.dump(self.result_dict, f, indent=4)

    def evaluate(self, save_file_path: str = './result.json') -> None:
        self.calculate_scores()
        self.save_result_as_json(file_path=save_file_path)
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
            'Rules'    : {'violation': 0, 'total': 0},
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
        # conf_matrix = confusion_matrix(y_test, y_pred_interpreted)
        precision = precision_score(y_test, y_pred_interpreted)
        recall = recall_score(y_test, y_pred_interpreted)
        f1 = f1_score(y_test, y_pred_interpreted)
        # class_report = classification_report(y_test, y_pred_interpreted)
        roc_auc = roc_auc_score(y_test, y_pred)

        self.result_dict['Accuracy'] = float(accuracy)
        # self.result_dict['Confusion_matrix'] = conf_matrix.tolist()
        self.result_dict['Precision'] = float(precision)
        self.result_dict['Recall'] = float(recall)
        self.result_dict['F1-score'] = float(f1)
        # self.result_dict['Classification_report'] = class_report
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


        print(y_pred_interpreted)


        for i, rule in enumerate(rules_tmp):
            outcome = rule["Outcome"]
            condition = " & ".join([f"{column} == {value}" for column, value in rule.items() if column != "Outcome"])
            tmp = y_pred_interpreted.loc[X_test.query(condition).index]
            violation_tmp = int((tmp != outcome).sum().iloc[0])
            self.result_dict['Rules']['violation'] += violation_tmp
            self.result_dict['Rules']['total'] += tmp.shape[0]
            self.result_dict['Rules_detail'][i] = {
                'violation': violation_tmp,
                'total': tmp.shape[0]
            }

    def save_result_as_json(self, file_path) -> None:
        with open(file_path, 'w') as f:
            json.dump(self.result_dict, f, indent=4)

    def evaluate(self, save_file_path: str = './result.json') -> None:
        self.calculate_scores()
        self.save_result_as_json(file_path=save_file_path)

    # def _judge_violations_pointwise(self) -> List[bool]:

    #     judgements_pointwise = []

    #     for j, (p_name, p) in enumerate(self.obj.predicates_dict.items()):
    #         for l in range(self.obj.len_l):
    #             x = self.obj.L[p_name][l, :-1]
    #             y = self.obj.L[p_name][l, -1]
    #             xi = self.obj.xi_jl[j, l]

    #             if xi.value is None:
    #                 xi = 0
    #             else:
    #                 xi = xi.value

    #             LHS = (y * (2 * p(x) - 1)).value
    #             RHS = 1 - 2 * xi

    #             judgements_pointwise += [
    #                 LHS >= RHS
    #             ]
        
    #     return judgements_pointwise

    # def _judge_violations_logical(self) -> List[bool]:
        
    #     judgements_logical = []

    #     # 仮
    #     U = next(iter(self.obj.U.values()))

    #     for u in U:
    #         KB_tmp = self.obj._calc_KB_at_datum(self.obj.KB, u)           

    #         for h, formula in enumerate(KB_tmp):
          
    #             xi = self.obj.xi_h[h, 0]
    #             if xi.value is None:
    #                 xi = 0
    #             else:
    #                 xi = xi.value

    #             formula_tmp = 0
    #             for item in formula:
    #                 if not is_symbol(item):
    #                     formula_tmp += item
    #             formula_tmp = negation(formula_tmp)
    #             formula_tmp = formula_tmp.value


    #             judgements_logical += [
    #                 0 <= xi,
    #                 formula_tmp <= xi
    #             ]
        
    #     return judgements_logical

    # def _judge_violations_consistency(self) -> List[bool]:
        
    #     judgements_consistency = []
    #     for (p_name, p) in self.obj.predicates_dict.items():
    #         for s in range(self.obj.len_s):
    #             x = self.obj.S[p_name][s]

    #             p_x = p(x).value

    #             judgements_consistency += [
    #                 p_x >= 0,
    #                 p_x <= 1
    #             ]
        
    #     return judgements_consistency

    # def calculate_violation(self) -> None:
    #     judgements_pointwise = self._judge_violations_pointwise()
    #     judgements_logical = self._judge_violations_logical()
    #     judgements_consistency = self._judge_violations_consistency()
    #     judgements = judgements_pointwise + judgements_logical + judgements_consistency

    #     violations_info = {
    #         'pointwise': judgements_pointwise,
    #         'logical': judgements_logical,
    #         'consistency': judgements_consistency,
    #         'all': judgements
    #     }

    #     for key, value in violations_info.items():
    #         key = 'VN' + '_' + key
    #         self.result_dict[key] = float(len(value) - sum(value))

    #     for key, value in violations_info.items():
    #         key = 'VR' + '_' + key
    #         self.result_dict[key] = float((len(value) - sum(value)) / len(value))
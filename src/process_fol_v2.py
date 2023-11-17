import sympy as sp

from .operators import negation
from .operators import Semantisize_symbols
from .misc import is_symbol, process_neg

# symbols_1 = ['¬', '∧', '∨', '⊗', '⊕', '→']

symbols_tmp = Semantisize_symbols()
symbols_1_semanticized = symbols_tmp.symbols_1_semanticized
symbols_3_semanticized = symbols_tmp.symbols_3_semanticized
symbols = list(symbols_1_semanticized.keys()) + list(symbols_3_semanticized.keys())


class FOLConverter:

    def __init__(self, file_path):
        self.file_path = file_path
        self.KB_origin = self._construct_KB()
        self.KB = None
        self.KB_tmp = None

        self.predicates_dict_tmp = None
        
    def _construct_KB(self):
        """
        KB の .txt ファイルを読み込む
        """    
        KB = []

        with open(self.file_path, 'r') as file:
            for line in file:
                formula = line.split()
                KB.append(formula)

        return KB     
    
    def _identify_predicates(self, KB):
        """
        KB 内の述語を特定し，
        各述語の係数を取り出すために
        sympy.Symbol で表現する
        """
        predicates = []

        for formula in KB:
            for item in formula:
                if item not in symbols and item not in predicates:
                    predicates.append(item)
        
        predicates_dict = {predicate: sp.Symbol(predicate) for predicate in predicates}

        return predicates_dict

    def _check_implication(self, formula):
        """
        formula (リスト) について，
        含意記号 '→' の数を調べ，
        その数が 1 以下になっているかを確認する
        """
        
        # 実質，ここには 1 つのインデックスしか入らない
        implication_idxs = []

        for i, item in enumerate(formula):
            if item == '→':
                implication_idxs.append(i)
        
        implication_num = len(implication_idxs)
        
        if implication_num == 0:
            return False, None
        elif implication_num == 1:
            implication_idx = implication_idxs[0]
            return True, implication_idx
        else:
            print('this formula may be invalid')

    def _eliminate_implication(self, formula):
        """
        formula (リスト) 内に含意記号 '→' あれば変換し，消去する 
        """
        implication_flag, target_idx = self._check_implication(formula)

        if implication_flag:
            # 含意記号 '→' を境に formula (list) を 2 つに分ける
            x = formula[:target_idx]
            y = formula[target_idx + 1:]

            # x → y = ¬ x ⊕ y
            x_new = negation(x)
            y_new = y
            new_operator = ['⊕']

            new_formula = x_new + new_operator + y_new
        else:
            new_formula = formula

        return new_formula
    
    def _drop_x(self, formula):
        """
        formula (リスト) 内の '(x)' を削除する
        """
        new_formula = []
        for item in formula:
            new_item = item.replace("(x)", "")
            new_formula.append(new_item)

        return new_formula
       
    def _get_idx_list(self, formula):
        """
        formula (リスト) 内の '¬' のインデックスリストと
        '¬' 以外のインデックスリストを返す
        """
        neg_idxs = []
        not_neg_idxs = []

        for i, item in enumerate(formula):
            if item == '¬':
                neg_idxs.append(i)
            else:
                not_neg_idxs.append(i)
        
        return neg_idxs, not_neg_idxs

    def _split_idx_list(self, idx_list):
        """
        インデックスリストを連続する部分リストに分割する
        """
        result = []
        tmp = []

        for i in range(len(idx_list)):
            if not tmp or idx_list[i] == tmp[-1] + 1:
                tmp.append(idx_list[i])
            else:
                result.append(tmp)
                tmp = [idx_list[i]]

        if tmp:
            result.append(tmp)

        return result
    
    def _eliminate_multi_negations(self, formula):
        """
        formula (リスト) 内の連続する '¬' を削除する
        """
        neg_idxs, not_neg_idxs = self._get_idx_list(formula)
        neg_idxs_decomposed = self._split_idx_list(neg_idxs)

        neg_idxs_new = []
        for tmp in neg_idxs_decomposed:
            if len(tmp) % 2 == 0:
                pass
            else:
                neg_idxs_new.append(tmp[0])
        
        idxs_new = sorted(neg_idxs_new + not_neg_idxs)
        
        formula_new = []
        for idx in idxs_new:
            item = formula[idx]
            formula_new.append(item)

        return formula_new

    
    def main(self):
        """
        KB,
        KB_tmp,
        predicates_dict

        をそれぞれ計算する
        """
        self.KB = []
        for formula in self.KB_origin:
            new_formula = self._eliminate_multi_negations(formula)
            new_formula = self._eliminate_implication(new_formula)
            new_formula = self._drop_x(new_formula)
            self.KB.append(new_formula)

        self.predicates_dict_tmp = self._identify_predicates(self.KB)

        self.KB_tmp = []
        for formula in self.KB:

            tmp_formula = []
            for item in formula:
                if item in self.predicates_dict_tmp.keys():
                    tmp_formula.append(self.predicates_dict_tmp[item])
                else:
                    tmp_formula.append(item)
                
            process_neg(tmp_formula)

            phi_h = []
            new_formula_1 = [sp.Symbol('1')]
            new_formula_2 = []

            tmp_new_formula_2 = 0
            for item in tmp_formula:
                if not is_symbol(item):
                    tmp_new_formula_2 += item
            
            new_formula_2.append(tmp_new_formula_2)

            phi_h.append(new_formula_1)
            phi_h.append(new_formula_2)
        
            self.KB_tmp.append(phi_h)


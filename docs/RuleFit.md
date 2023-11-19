# 注意事項

2023/11/18 日の時点で，[RuleFit](https://github.com/christophM/rulefit/tree/master) ライブラリの RuleFit クラスは生成したルールを一覧で取得することができなくなっています．

そのため，RuleFit クラスに以下の get_rules メソッドを追加する必要があります．

```
    def get_rules(rf, exclude_zero_coef=False, subregion=None):
        """Return the estimated rules
        Parameters
        ----------
        exclude_zero_coef: If True (default), returns only the rules with an estimated
                           coefficient not equalt to  zero.
        subregion: If None (default) returns global importances (FP 2004 eq. 28/29), else returns importance over
                           subregion of inputs (FP 2004 eq. 30/31/32).
        Returns
        -------
        rules: pandas.DataFrame with the rules. Column 'rule' describes the rule, 'coef' holds
               the coefficients and 'support' the support of the rule in the training
               data set (X)
        """
        n_features= len(rf.coef_) - len(rf.rule_ensemble.rules)
        rule_ensemble = list(rf.rule_ensemble.rules)
        output_rules = []
        ## Add coefficients for linear effects
        for i in range(0, n_features):
            if rf.lin_standardise:
                coef=rf.coef_[i]*rf.friedscale.scale_multipliers[i]
            else:
                coef=rf.coef_[i]
            if subregion is None:
                importance = abs(coef)*rf.stddev[i]
            else:
                subregion = np.array(subregion)
                importance = sum(abs(coef)* abs([ x[i] for x in rf.winsorizer.trim(subregion) ] - rf.mean[i]))/len(subregion)
            output_rules += [(rf.feature_names[i], 'linear',coef, 1, importance)]
        ## Add rules
        for i in range(0, len(rf.rule_ensemble.rules)):
            rule = rule_ensemble[i]
            coef=rf.coef_[i + n_features]
            if subregion is None:
                importance = abs(coef)*(rule.support * (1-rule.support))**(1/2)
            else:
                rkx = rule.transform(subregion)
                importance = sum(abs(coef) * abs(rkx - rule.support))/len(subregion)

            output_rules += [(rule.__str__(), 'rule', coef,  rule.support, importance)]
        rules = pd.DataFrame(output_rules, columns=["rule", "type","coef", "support", "importance"])
        if exclude_zero_coef:
            rules = rules.ix[rules.coef != 0]
        return rules
```


# 参照

1. [RuleFit ライブラリ GitHub](https://github.com/christophM/rulefit/blob/master/rulefit/rulefit.py)
2. [get_rules メソッド](https://github.com/christophM/rulefit/pull/56/commits/e04fc1e70555350285fce18fd8a3b86e9a2bfd02)
3. [Example コード](https://github.com/christophM/rulefit/blob/master/example_simulated.py)
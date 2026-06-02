# arousal_score_audio_gradient_boosting_selected

rows: 40
features: 74
split: holdout
cv: KFold(n_splits=3, random_state=42, shuffle=True)
grid_preset: quick
grid_candidates: 48
search_fit_time: 2.5s

## Selected model

selected_model: baseline
selection_rule: lower holdout MAE
selected_holdout_mae: 0.143
selected_holdout_r2: 0.168

## Baseline default GradientBoostingRegressor

mae: 0.143
r2: 0.168

## Tuned GradientBoostingRegressor candidate

best_params:
```python
{'model__learning_rate': 0.03, 'model__max_depth': 1, 'model__min_samples_leaf': 2, 'model__n_estimators': 50, 'model__subsample': 0.8}
```
best_cv_mae: 0.141
holdout_mae: 0.165
holdout_r2: -0.244

## Top grid results

```text
 rank_test_mae  mean_test_mae  std_test_mae  mean_train_mae  std_train_mae  param_model__learning_rate  param_model__max_depth  param_model__min_samples_leaf  param_model__n_estimators  param_model__subsample
             1      -0.141063      0.020690       -0.073991       0.014689                        0.03                       1                              2                         50                     0.8
             2      -0.143377      0.023399       -0.073936       0.015344                        0.03                       1                              4                         50                     0.8
             3      -0.147311      0.023148       -0.052303       0.008819                        0.03                       2                              4                         50                     0.8
             4      -0.150171      0.019612       -0.049715       0.009206                        0.03                       2                              2                         50                     0.8
             5      -0.150358      0.019680       -0.077281       0.014803                        0.03                       1                              4                         50                     1.0
             6      -0.150385      0.022481       -0.049753       0.011662                        0.03                       1                              2                        100                     0.8
             7      -0.151046      0.020683       -0.076927       0.014672                        0.03                       1                              2                         50                     1.0
             8      -0.151199      0.016850       -0.046011       0.009458                        0.03                       2                              2                         50                     1.0
             9      -0.153111      0.021557       -0.052914       0.012968                        0.03                       1                              4                        100                     0.8
            10      -0.153368      0.018020       -0.049847       0.009994                        0.03                       2                              4                         50                     1.0
            11      -0.154437      0.025170       -0.055787       0.011754                        0.03                       1                              2                        100                     1.0
            12      -0.155868      0.020751       -0.058609       0.013255                        0.03                       1                              4                        100                     1.0
```

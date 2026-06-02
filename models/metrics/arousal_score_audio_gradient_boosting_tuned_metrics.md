# arousal_score_audio_gradient_boosting_tuned

rows: 40
features: 74
split: holdout
cv: KFold(n_splits=3, random_state=42, shuffle=True)
grid_candidates: 576
fit_time: 1m 15.2s

## Best parameters

```python
{'model__learning_rate': 0.1, 'model__max_depth': 1, 'model__min_samples_leaf': 8, 'model__n_estimators': 400, 'model__subsample': 0.6}
```

## Cross-validation result

best_cv_mae: 0.127
best_cv_neg_mae: -0.127

## Holdout result

mae: 0.195
r2: -0.659

## Top grid results

```text
 rank_test_mae  mean_test_mae  std_test_mae  mean_train_mae  std_train_mae  param_model__learning_rate  param_model__max_depth  param_model__min_samples_leaf  param_model__n_estimators  param_model__subsample
             1      -0.127273      0.022508       -0.121884       0.011890                        0.10                       3                              8                        400                     0.6
             1      -0.127273      0.022508       -0.121884       0.011890                        0.10                       2                              8                        400                     0.6
             1      -0.127273      0.022508       -0.121884       0.011890                        0.10                       1                              8                        400                     0.6
             4      -0.127982      0.021432       -0.121853       0.011318                        0.10                       3                              8                         50                     0.6
             4      -0.127982      0.021432       -0.121853       0.011318                        0.10                       2                              8                         50                     0.6
             4      -0.127982      0.021432       -0.121853       0.011318                        0.10                       1                              8                         50                     0.6
             7      -0.128008      0.022209       -0.121577       0.011903                        0.05                       3                              8                        400                     0.6
             7      -0.128008      0.022209       -0.121577       0.011903                        0.05                       2                              8                        400                     0.6
             7      -0.128008      0.022209       -0.121577       0.011903                        0.05                       1                              8                        400                     0.6
            10      -0.128018      0.021767       -0.121693       0.011566                        0.05                       3                              8                         50                     0.6
            10      -0.128018      0.021767       -0.121693       0.011566                        0.05                       2                              8                         50                     0.6
            10      -0.128018      0.021767       -0.121693       0.011566                        0.05                       1                              8                         50                     0.6
            13      -0.128307      0.021803       -0.121562       0.011698                        0.03                       1                              8                         50                     0.6
            13      -0.128307      0.021803       -0.121562       0.011698                        0.03                       3                              8                         50                     0.6
            13      -0.128307      0.021803       -0.121562       0.011698                        0.03                       2                              8                         50                     0.6
```

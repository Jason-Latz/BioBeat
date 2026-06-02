# valence_label_audio_gradient_boosting_tuned

rows: 229
features: 74
class_counts: {'neutral': 102, 'negative': 65, 'positive': 62}
split: holdout
cv: StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
grid_candidates: 576
fit_time: 28m 51.0s

## Best parameters

```python
{'model__learning_rate': 0.1, 'model__max_depth': 2, 'model__min_samples_leaf': 1, 'model__n_estimators': 400, 'model__subsample': 1.0}
```

## Cross-validation result

best_cv_macro_f1: 0.523

## Holdout result

accuracy: 0.586
balanced_accuracy: 0.588
macro_f1: 0.588
confusion_matrix:
[[10  5  1]
 [ 8 15  3]
 [ 2  5  9]]
classification_report:
              precision    recall  f1-score   support

    negative       0.50      0.62      0.56        16
     neutral       0.60      0.58      0.59        26
    positive       0.69      0.56      0.62        16

    accuracy                           0.59        58
   macro avg       0.60      0.59      0.59        58
weighted avg       0.60      0.59      0.59        58


## Top grid results

```text
 rank_test_macro_f1  mean_test_macro_f1  std_test_macro_f1  mean_train_macro_f1  std_train_macro_f1  param_model__learning_rate  param_model__max_depth  param_model__min_samples_leaf  param_model__n_estimators  param_model__subsample
                  1            0.522962           0.074722             0.943012            0.014274                        0.10                       2                              1                        400                     1.0
                  2            0.510492           0.081394             0.822825            0.016816                        0.01                       2                              2                        100                     0.8
                  3            0.506951           0.086250             0.873828            0.012653                        0.03                       2                              2                         50                     0.8
                  4            0.505831           0.118992             0.937096            0.013576                        0.03                       3                              2                         50                     0.8
                  5            0.505620           0.079593             0.824710            0.015994                        0.01                       2                              1                        100                     0.8
                  6            0.505168           0.089688             0.907589            0.008852                        0.01                       2                              2                        200                     0.8
                  7            0.500525           0.074050             0.834133            0.009321                        0.03                       2                              8                         50                     1.0
                  8            0.500419           0.043592             0.941583            0.016712                        0.05                       2                              1                        100                     1.0
                  9            0.499402           0.078586             0.913325            0.010365                        0.01                       2                              1                        200                     0.8
                 10            0.497766           0.100165             0.925595            0.008017                        0.03                       1                              1                        400                     1.0
                 11            0.497766           0.084685             0.908650            0.016812                        0.05                       1                              1                        200                     1.0
                 12            0.496500           0.086578             0.766722            0.008974                        0.03                       1                              2                        100                     1.0
                 12            0.496500           0.086578             0.766722            0.008974                        0.03                       1                              1                        100                     1.0
                 14            0.496438           0.093522             0.925838            0.009077                        0.03                       1                              4                        400                     1.0
                 15            0.495000           0.073041             0.923403            0.009471                        0.03                       2                              1                        100                     1.0
```

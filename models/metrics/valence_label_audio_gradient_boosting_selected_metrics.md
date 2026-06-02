# valence_label_audio_gradient_boosting_selected

rows: 229
features: 74
class_counts: {'neutral': 102, 'negative': 65, 'positive': 62}
split: holdout
cv: StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
grid_preset: quick
grid_candidates: 48
search_fit_time: 11.3s

## Selected model

selected_model: baseline
selection_rule: higher holdout macro F1
selected_accuracy: 0.638
selected_balanced_accuracy: 0.627
selected_macro_f1: 0.634
confusion_matrix:
[[ 9  6  1]
 [ 5 18  3]
 [ 1  5 10]]
classification_report:
              precision    recall  f1-score   support

    negative       0.60      0.56      0.58        16
     neutral       0.62      0.69      0.65        26
    positive       0.71      0.62      0.67        16

    accuracy                           0.64        58
   macro avg       0.64      0.63      0.63        58
weighted avg       0.64      0.64      0.64        58


## Baseline default GradientBoostingClassifier

accuracy: 0.638
balanced_accuracy: 0.627
macro_f1: 0.634

## Tuned GradientBoostingClassifier candidate

best_params:
```python
{'model__learning_rate': 0.03, 'model__max_depth': 2, 'model__min_samples_leaf': 2, 'model__n_estimators': 50, 'model__subsample': 0.8}
```
best_cv_macro_f1: 0.507
holdout_accuracy: 0.534
holdout_balanced_accuracy: 0.502
holdout_macro_f1: 0.512

## Top grid results

```text
 rank_test_macro_f1  mean_test_macro_f1  std_test_macro_f1  mean_train_macro_f1  std_train_macro_f1  param_model__learning_rate  param_model__max_depth  param_model__min_samples_leaf  param_model__n_estimators  param_model__subsample
                  1            0.506951           0.086250             0.873828            0.012653                        0.03                       2                              2                         50                     0.8
                  2            0.496500           0.086578             0.766722            0.008974                        0.03                       1                              2                        100                     1.0
                  3            0.493456           0.083004             0.928670            0.011640                        0.03                       2                              2                        100                     0.8
                  4            0.488244           0.093278             0.911527            0.012628                        0.10                       1                              4                        100                     1.0
                  5            0.486635           0.100406             0.863704            0.011637                        0.10                       1                              4                         50                     0.8
                  6            0.486618           0.072146             0.778211            0.017869                        0.03                       1                              2                        100                     0.8
                  7            0.486543           0.065857             0.866292            0.009969                        0.03                       1                              2                        200                     1.0
                  8            0.485835           0.094452             0.875881            0.016571                        0.03                       1                              4                        200                     0.8
                  9            0.483445           0.088149             0.759793            0.012412                        0.03                       1                              4                        100                     1.0
                 10            0.483405           0.124960             0.943560            0.014021                        0.10                       1                              2                        200                     1.0
                 11            0.483130           0.080981             0.925655            0.014049                        0.10                       1                              4                        100                     0.8
                 12            0.482990           0.078290             0.856168            0.020015                        0.03                       1                              4                        200                     1.0
```

# arousal_score_audio_mlp_tuned

rows: 40
features: 74
split: holdout
cv: KFold(n_splits=2, random_state=42, shuffle=True)
grid_candidates: 360

## Best parameters

```python
{'model__activation': 'relu', 'model__alpha': 0.1, 'model__batch_size': 8, 'model__hidden_layer_sizes': (32,), 'model__learning_rate_init': 0.003}
```

## Cross-validation result

best_cv_mae: 0.159
best_cv_neg_mae: -0.159

## Holdout result

mae: 0.453
r2: -5.308

## Top grid results

```text
 rank  mean_cv_neg_mae  std_cv_neg_mae  mean_train_score  std_train_score param_model__hidden_layer_sizes param_model__activation  param_model__alpha  param_model__learning_rate_init  param_model__batch_size
    1        -0.158556        0.000845         -0.022499         0.013849                           (32,)                    relu                 0.1                           0.0030                        8
    2        -0.159066        0.007765         -0.033737         0.000560                        (32, 16)                    relu                 0.1                           0.0003                        8
    3        -0.160801        0.000733         -0.019395         0.015229                           (32,)                    relu                 0.1                           0.0003                        8
    4        -0.163684        0.000072         -0.031414         0.010779                        (32, 16)                    relu                 0.1                           0.0030                        8
    5        -0.166286        0.002935         -0.024959         0.016314                           (32,)                    relu                 0.1                           0.0030                       32
    5        -0.166286        0.002935         -0.024959         0.016314                           (32,)                    relu                 0.1                           0.0030                       16
    7        -0.170641        0.005518         -0.028932         0.003563                        (32, 16)                    relu                 0.1                           0.0030                       32
    7        -0.170641        0.005518         -0.028932         0.003563                        (32, 16)                    relu                 0.1                           0.0030                       16
    9        -0.181010        0.016473         -0.034799         0.003998                        (32, 16)                    relu                 0.1                           0.0010                        8
   10        -0.214081        0.051606         -0.028333         0.012681                        (32, 16)                    tanh                 0.1                           0.0003                        8
```

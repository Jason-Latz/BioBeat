# arousal_binary

rows: 144
positive_rate: 0.278
split: holdout

## Logistic regression

accuracy: 0.889
confusion_matrix:
[[23  3]
 [ 1  9]]
classification_report:
              precision    recall  f1-score   support

           0       0.96      0.88      0.92        26
           1       0.75      0.90      0.82        10

    accuracy                           0.89        36
   macro avg       0.85      0.89      0.87        36
weighted avg       0.90      0.89      0.89        36


## Random forest

accuracy: 0.972
confusion_matrix:
[[26  0]
 [ 1  9]]
classification_report:
              precision    recall  f1-score   support

           0       0.96      1.00      0.98        26
           1       1.00      0.90      0.95        10

    accuracy                           0.97        36
   macro avg       0.98      0.95      0.96        36
weighted avg       0.97      0.97      0.97        36


## Random forest feature importance

                 feature  importance
  spectral_centroid_mean    0.105486
       eda_max_amplitude    0.076568
                hr_slope    0.064944
   spectral_rolloff_mean    0.062094
               eda_slope    0.054060
             mfcc_mean_3    0.053830
 hr_change_from_baseline    0.053364
 zero_crossing_rate_mean    0.051742
            eda_recovery    0.043776
eda_change_from_baseline    0.042763
             hr_recovery    0.038270
             mfcc_mean_1    0.022627
 spectral_bandwidth_mean    0.022121
           chroma_mean_3    0.021329
             mfcc_mean_4    0.018177

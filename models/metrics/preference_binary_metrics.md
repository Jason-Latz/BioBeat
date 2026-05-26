# preference_binary

rows: 96
positive_rate: 0.354
split: holdout

## Logistic regression

accuracy: 0.750
confusion_matrix:
[[13  3]
 [ 3  5]]
classification_report:
              precision    recall  f1-score   support

           0       0.81      0.81      0.81        16
           1       0.62      0.62      0.62         8

    accuracy                           0.75        24
   macro avg       0.72      0.72      0.72        24
weighted avg       0.75      0.75      0.75        24


## Random forest

accuracy: 0.542
confusion_matrix:
[[12  4]
 [ 7  1]]
classification_report:
              precision    recall  f1-score   support

           0       0.63      0.75      0.69        16
           1       0.20      0.12      0.15         8

    accuracy                           0.54        24
   macro avg       0.42      0.44      0.42        24
weighted avg       0.49      0.54      0.51        24


## Random forest feature importance

                 feature  importance
                 hr_mean    0.069806
            eda_recovery    0.068696
                  hr_max    0.067685
                eda_mean    0.063054
               eda_slope    0.063011
 hr_change_from_baseline    0.061666
       eda_max_amplitude    0.059188
             hr_recovery    0.054512
                hr_slope    0.052739
eda_change_from_baseline    0.051903
          eda_peak_count    0.031175
    spectral_rolloff_std    0.016743
            mfcc_mean_13    0.013622
             mfcc_mean_1    0.013000
           chroma_mean_1    0.011303

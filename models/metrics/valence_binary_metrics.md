# valence_binary

rows: 96
positive_rate: 0.510
split: holdout

## Logistic regression

accuracy: 0.708
confusion_matrix:
[[8 4]
 [3 9]]
classification_report:
              precision    recall  f1-score   support

           0       0.73      0.67      0.70        12
           1       0.69      0.75      0.72        12

    accuracy                           0.71        24
   macro avg       0.71      0.71      0.71        24
weighted avg       0.71      0.71      0.71        24


## Random forest

accuracy: 0.750
confusion_matrix:
[[ 8  4]
 [ 2 10]]
classification_report:
              precision    recall  f1-score   support

           0       0.80      0.67      0.73        12
           1       0.71      0.83      0.77        12

    accuracy                           0.75        24
   macro avg       0.76      0.75      0.75        24
weighted avg       0.76      0.75      0.75        24


## Random forest feature importance

                 feature  importance
            mfcc_mean_13    0.062477
             hr_recovery    0.053746
 hr_change_from_baseline    0.050143
             mfcc_mean_7    0.042906
              mfcc_std_8    0.040081
                eda_mean    0.036882
                  hr_max    0.034628
               eda_slope    0.032337
eda_change_from_baseline    0.031128
                hr_slope    0.029422
            eda_recovery    0.027869
              mfcc_std_6    0.026805
       eda_max_amplitude    0.026485
  spectral_bandwidth_std    0.025609
             mfcc_std_11    0.024024

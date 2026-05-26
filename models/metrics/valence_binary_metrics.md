# valence_binary

rows: 144
positive_rate: 0.556
split: holdout

## Logistic regression

accuracy: 0.694
confusion_matrix:
[[ 7  9]
 [ 2 18]]
classification_report:
              precision    recall  f1-score   support

           0       0.78      0.44      0.56        16
           1       0.67      0.90      0.77        20

    accuracy                           0.69        36
   macro avg       0.72      0.67      0.66        36
weighted avg       0.72      0.69      0.67        36


## Random forest

accuracy: 0.667
confusion_matrix:
[[ 5 11]
 [ 1 19]]
classification_report:
              precision    recall  f1-score   support

           0       0.83      0.31      0.45        16
           1       0.63      0.95      0.76        20

    accuracy                           0.67        36
   macro avg       0.73      0.63      0.61        36
weighted avg       0.72      0.67      0.62        36


## Random forest feature importance

                 feature  importance
             mfcc_mean_7    0.054049
              mfcc_std_7    0.048980
                eda_mean    0.045441
eda_change_from_baseline    0.038609
            mfcc_mean_13    0.035332
       eda_max_amplitude    0.034930
          eda_peak_count    0.033394
             hr_recovery    0.031873
                hr_slope    0.031389
               eda_slope    0.031235
 hr_change_from_baseline    0.030255
                 hr_mean    0.028205
            eda_recovery    0.025174
                  hr_max    0.024267
              mfcc_std_5    0.023662

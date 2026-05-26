# preference_binary

rows: 144
positive_rate: 0.382
split: holdout

## Logistic regression

accuracy: 0.583
confusion_matrix:
[[16  6]
 [ 9  5]]
classification_report:
              precision    recall  f1-score   support

           0       0.64      0.73      0.68        22
           1       0.45      0.36      0.40        14

    accuracy                           0.58        36
   macro avg       0.55      0.54      0.54        36
weighted avg       0.57      0.58      0.57        36


## Random forest

accuracy: 0.528
confusion_matrix:
[[17  5]
 [12  2]]
classification_report:
              precision    recall  f1-score   support

           0       0.59      0.77      0.67        22
           1       0.29      0.14      0.19        14

    accuracy                           0.53        36
   macro avg       0.44      0.46      0.43        36
weighted avg       0.47      0.53      0.48        36


## Random forest feature importance

                 feature  importance
eda_change_from_baseline    0.051866
                  hr_max    0.046162
               eda_slope    0.045643
             hr_recovery    0.045563
 hr_change_from_baseline    0.044819
                hr_slope    0.044042
            eda_recovery    0.043490
       eda_max_amplitude    0.042534
                eda_mean    0.041698
                 hr_mean    0.039429
          eda_peak_count    0.033439
             mfcc_std_12    0.027279
             mfcc_mean_7    0.026735
            mfcc_mean_13    0.024630
  zero_crossing_rate_std    0.018344

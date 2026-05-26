# arousal_binary

rows: 96
positive_rate: 0.406
split: holdout

## Logistic regression

accuracy: 0.875
confusion_matrix:
[[14  0]
 [ 3  7]]
classification_report:
              precision    recall  f1-score   support

           0       0.82      1.00      0.90        14
           1       1.00      0.70      0.82        10

    accuracy                           0.88        24
   macro avg       0.91      0.85      0.86        24
weighted avg       0.90      0.88      0.87        24


## Random forest

accuracy: 0.875
confusion_matrix:
[[14  0]
 [ 3  7]]
classification_report:
              precision    recall  f1-score   support

           0       0.82      1.00      0.90        14
           1       1.00      0.70      0.82        10

    accuracy                           0.88        24
   macro avg       0.91      0.85      0.86        24
weighted avg       0.90      0.88      0.87        24


## Random forest feature importance

                 feature  importance
 zero_crossing_rate_mean    0.115275
  spectral_centroid_mean    0.106811
             mfcc_mean_1    0.071475
       eda_max_amplitude    0.064939
           chroma_mean_3    0.061689
   spectral_rolloff_mean    0.060008
eda_change_from_baseline    0.052564
         rms_energy_mean    0.042727
             mfcc_mean_2    0.040339
            eda_recovery    0.039070
           chroma_mean_4    0.029665
 hr_change_from_baseline    0.025490
            mfcc_mean_12    0.024235
            mfcc_mean_13    0.023303
               eda_slope    0.020740

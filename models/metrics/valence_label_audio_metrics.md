# valence_label_audio

rows: 229
class_counts: {'neutral': 102, 'negative': 65, 'positive': 62}
split: holdout

## Logistic Regression

accuracy: 0.466
confusion_matrix:
[[ 7  8  1]
 [10 12  4]
 [ 2  6  8]]
classification_report:
              precision    recall  f1-score   support

    negative       0.37      0.44      0.40        16
     neutral       0.46      0.46      0.46        26
    positive       0.62      0.50      0.55        16

    accuracy                           0.47        58
   macro avg       0.48      0.47      0.47        58
weighted avg       0.48      0.47      0.47        58


## Random Forest Classifier

accuracy: 0.552
confusion_matrix:
[[ 8  5  3]
 [ 6 15  5]
 [ 1  6  9]]
classification_report:
              precision    recall  f1-score   support

    negative       0.53      0.50      0.52        16
     neutral       0.58      0.58      0.58        26
    positive       0.53      0.56      0.55        16

    accuracy                           0.55        58
   macro avg       0.55      0.55      0.55        58
weighted avg       0.55      0.55      0.55        58


## Extra Trees Classifier

accuracy: 0.569
confusion_matrix:
[[ 8  5  3]
 [ 5 16  5]
 [ 1  6  9]]
classification_report:
              precision    recall  f1-score   support

    negative       0.57      0.50      0.53        16
     neutral       0.59      0.62      0.60        26
    positive       0.53      0.56      0.55        16

    accuracy                           0.57        58
   macro avg       0.56      0.56      0.56        58
weighted avg       0.57      0.57      0.57        58


## Gradient Boosting Classifier

accuracy: 0.621
confusion_matrix:
[[ 9  5  2]
 [ 6 17  3]
 [ 1  5 10]]
classification_report:
              precision    recall  f1-score   support

    negative       0.56      0.56      0.56        16
     neutral       0.63      0.65      0.64        26
    positive       0.67      0.62      0.65        16

    accuracy                           0.62        58
   macro avg       0.62      0.61      0.62        58
weighted avg       0.62      0.62      0.62        58


## Svc Rbf

accuracy: 0.534
confusion_matrix:
[[ 7  5  4]
 [ 7 14  5]
 [ 3  3 10]]
classification_report:
              precision    recall  f1-score   support

    negative       0.41      0.44      0.42        16
     neutral       0.64      0.54      0.58        26
    positive       0.53      0.62      0.57        16

    accuracy                           0.53        58
   macro avg       0.52      0.53      0.53        58
weighted avg       0.54      0.53      0.54        58


## Knn Classifier

accuracy: 0.552
confusion_matrix:
[[ 9  4  3]
 [ 7 14  5]
 [ 1  6  9]]
classification_report:
              precision    recall  f1-score   support

    negative       0.53      0.56      0.55        16
     neutral       0.58      0.54      0.56        26
    positive       0.53      0.56      0.55        16

    accuracy                           0.55        58
   macro avg       0.55      0.55      0.55        58
weighted avg       0.55      0.55      0.55        58


## Mlp Classifier

accuracy: 0.534
confusion_matrix:
[[ 8  7  1]
 [ 7 15  4]
 [ 1  7  8]]
classification_report:
              precision    recall  f1-score   support

    negative       0.50      0.50      0.50        16
     neutral       0.52      0.58      0.55        26
    positive       0.62      0.50      0.55        16

    accuracy                           0.53        58
   macro avg       0.54      0.53      0.53        58
weighted avg       0.54      0.53      0.53        58


## Random forest feature importance

                feature  importance
    onset_strength_mean    0.039161
     onset_strength_std    0.031865
        loudness_db_std    0.026084
            mfcc_mean_8    0.022999
             mfcc_std_6    0.021631
      loudness_db_range    0.020156
            mfcc_mean_7    0.019413
 spectral_centroid_mean    0.018889
            mfcc_mean_3    0.018738
  major_key_correlation    0.018596
          chroma_mean_8    0.018563
              beat_rate    0.018335
spectral_centroid_slope    0.017158
           mfcc_mean_11    0.016999
           mfcc_mean_13    0.016598

## Extra trees feature importance

               feature  importance
    onset_strength_std    0.023626
   onset_strength_mean    0.022203
           mfcc_mean_3    0.019686
           mfcc_mean_8    0.019352
     loudness_db_range    0.017433
                 tempo    0.016966
zero_crossing_rate_std    0.016904
        rms_energy_std    0.016665
            mfcc_std_6    0.016640
 major_key_correlation    0.016577
           mfcc_mean_1    0.016123
           mfcc_mean_7    0.016044
            mfcc_std_3    0.015998
            mfcc_std_8    0.015964
            beat_count    0.015960

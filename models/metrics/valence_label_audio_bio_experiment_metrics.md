# valence_label_audio_bio_experiment

rows: 229
class_counts: {'neutral': 102, 'negative': 65, 'positive': 62}
split: holdout

## Logistic Regression

accuracy: 0.414
confusion_matrix:
[[ 6  8  2]
 [14 10  2]
 [ 2  6  8]]
classification_report:
              precision    recall  f1-score   support

    negative       0.27      0.38      0.32        16
     neutral       0.42      0.38      0.40        26
    positive       0.67      0.50      0.57        16

    accuracy                           0.41        58
   macro avg       0.45      0.42      0.43        58
weighted avg       0.45      0.41      0.42        58


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
[[ 9  5  2]
 [ 6 15  5]
 [ 1  6  9]]
classification_report:
              precision    recall  f1-score   support

    negative       0.56      0.56      0.56        16
     neutral       0.58      0.58      0.58        26
    positive       0.56      0.56      0.56        16

    accuracy                           0.57        58
   macro avg       0.57      0.57      0.57        58
weighted avg       0.57      0.57      0.57        58


## Gradient Boosting Classifier

accuracy: 0.603
confusion_matrix:
[[ 9  5  2]
 [ 6 17  3]
 [ 1  6  9]]
classification_report:
              precision    recall  f1-score   support

    negative       0.56      0.56      0.56        16
     neutral       0.61      0.65      0.63        26
    positive       0.64      0.56      0.60        16

    accuracy                           0.60        58
   macro avg       0.60      0.59      0.60        58
weighted avg       0.60      0.60      0.60        58


## Svc Rbf

accuracy: 0.534
confusion_matrix:
[[ 8  5  3]
 [ 7 14  5]
 [ 3  4  9]]
classification_report:
              precision    recall  f1-score   support

    negative       0.44      0.50      0.47        16
     neutral       0.61      0.54      0.57        26
    positive       0.53      0.56      0.55        16

    accuracy                           0.53        58
   macro avg       0.53      0.53      0.53        58
weighted avg       0.54      0.53      0.54        58


## Knn Classifier

accuracy: 0.517
confusion_matrix:
[[ 6  6  4]
 [ 5 15  6]
 [ 1  6  9]]
classification_report:
              precision    recall  f1-score   support

    negative       0.50      0.38      0.43        16
     neutral       0.56      0.58      0.57        26
    positive       0.47      0.56      0.51        16

    accuracy                           0.52        58
   macro avg       0.51      0.50      0.50        58
weighted avg       0.52      0.52      0.51        58


## Mlp Classifier

accuracy: 0.552
confusion_matrix:
[[ 7  5  4]
 [ 6 18  2]
 [ 1  8  7]]
classification_report:
              precision    recall  f1-score   support

    negative       0.50      0.44      0.47        16
     neutral       0.58      0.69      0.63        26
    positive       0.54      0.44      0.48        16

    accuracy                           0.55        58
   macro avg       0.54      0.52      0.53        58
weighted avg       0.55      0.55      0.55        58


## Random forest feature importance

                feature  importance
     onset_strength_std    0.034101
    onset_strength_mean    0.032743
            mfcc_mean_8    0.024146
        loudness_db_std    0.021709
      loudness_db_range    0.021599
 spectral_centroid_mean    0.019710
            mfcc_mean_3    0.019662
  major_key_correlation    0.019495
            mfcc_mean_7    0.019246
             mfcc_std_6    0.019158
          chroma_mean_3    0.017624
          chroma_mean_8    0.017012
spectral_centroid_slope    0.016674
           mfcc_mean_11    0.016476
            mfcc_mean_2    0.016105

## Extra trees feature importance

               feature  importance
    onset_strength_std    0.025020
   onset_strength_mean    0.022299
           mfcc_mean_7    0.017942
           mfcc_mean_3    0.017835
        rms_energy_std    0.017596
         chroma_mean_8    0.017399
           mfcc_mean_8    0.017367
 major_key_correlation    0.016514
          mfcc_mean_11    0.016289
     loudness_db_range    0.016280
       loudness_db_std    0.015921
           mfcc_mean_1    0.015915
            mfcc_std_8    0.015819
zero_crossing_rate_std    0.015467
            mfcc_std_6    0.015435

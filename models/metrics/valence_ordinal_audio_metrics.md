# valence_ordinal_audio

rows: 229
target_mean: -0.013
split: holdout

## Ridge Regression

mae: 0.682
r2: -0.166

## Random Forest Regressor

mae: 0.583
r2: 0.153

## Extra Trees Regressor

mae: 0.586
r2: 0.084

## Gradient Boosting Regressor

mae: 0.588
r2: 0.130

## Svr Rbf

mae: 0.568
r2: 0.190

## Knn Regressor

mae: 0.604
r2: -0.097

## Mlp Regressor

mae: 0.653
r2: -0.175

## Random forest feature importance

                feature  importance
    onset_strength_mean    0.071046
      loudness_db_range    0.053359
        loudness_db_std    0.044054
          chroma_mean_8    0.042111
         rms_energy_std    0.038754
      rhythm_regularity    0.026789
       rms_energy_range    0.026332
            mfcc_mean_3    0.024287
           mfcc_mean_11    0.023083
          chroma_mean_1    0.020847
            mfcc_mean_8    0.020035
            mfcc_mean_1    0.019149
        mode_confidence    0.017214
spectral_centroid_slope    0.016999
  major_key_correlation    0.016301

## Extra trees feature importance

               feature  importance
        rms_energy_std    0.022748
   onset_strength_mean    0.022451
      rms_energy_range    0.022348
            mfcc_std_8    0.020453
         chroma_mean_8    0.020364
           mfcc_std_11    0.019572
    onset_strength_std    0.019180
       loudness_db_std    0.018463
 spectral_rolloff_mean    0.018062
       mode_confidence    0.017250
spectral_centroid_mean    0.016685
zero_crossing_rate_std    0.016537
         chroma_mean_9    0.016377
     loudness_db_range    0.015937
                 tempo    0.015780

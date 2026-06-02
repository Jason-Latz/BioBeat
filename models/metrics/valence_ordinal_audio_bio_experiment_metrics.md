# valence_ordinal_audio_bio_experiment

rows: 229
target_mean: -0.013
split: holdout

## Ridge Regression

mae: 0.714
r2: -0.371

## Random Forest Regressor

mae: 0.589
r2: 0.142

## Extra Trees Regressor

mae: 0.608
r2: 0.060

## Gradient Boosting Regressor

mae: 0.590
r2: 0.140

## Svr Rbf

mae: 0.618
r2: 0.082

## Knn Regressor

mae: 0.669
r2: -0.227

## Mlp Regressor

mae: 0.701
r2: -0.378

## Random forest feature importance

                feature  importance
    onset_strength_mean    0.072653
      loudness_db_range    0.052734
        loudness_db_std    0.041894
          chroma_mean_8    0.040941
         rms_energy_std    0.038102
      rhythm_regularity    0.026439
       rms_energy_range    0.025699
            mfcc_mean_3    0.023813
           mfcc_mean_11    0.022784
          chroma_mean_1    0.022032
            mfcc_mean_1    0.019136
            mfcc_mean_8    0.018980
spectral_centroid_slope    0.017367
        mode_confidence    0.017355
  major_key_correlation    0.016314

## Extra trees feature importance

               feature  importance
        rms_energy_std    0.023341
   onset_strength_mean    0.022119
      rms_energy_range    0.021244
         chroma_mean_8    0.020764
            mfcc_std_8    0.020522
    onset_strength_std    0.019262
       loudness_db_std    0.018971
 spectral_rolloff_mean    0.017600
spectral_centroid_mean    0.017351
           mfcc_std_11    0.016896
           mfcc_std_12    0.016594
       mode_confidence    0.016580
           mfcc_std_10    0.016509
           mfcc_mean_2    0.016084
            mfcc_std_7    0.015610
